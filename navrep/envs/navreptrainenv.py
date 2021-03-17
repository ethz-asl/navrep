import configparser
import time
import gym
from gym import spaces
import numpy as np
from pandas import DataFrame
import tensorflow as tf
import threading
import os
from CMap2D import flatten_contours, render_contours_in_lidar, CMap2D, CSimAgent, fast_2f_norm
from pose2d import apply_tf_to_vel, inverse_pose2d, apply_tf_to_pose
from pkg_resources import resource_filename

import crowd_sim  # adds CrowdSim-v0 to gym  # noqa
from crowd_sim.envs.crowd_sim import CrowdSim  # reference to env code  # noqa
from crowd_sim.envs.utils.robot import Robot
from crowd_nav.policy.network_om import SDOADRL
from crowd_sim.envs.utils.action import ActionXYRot
from crowd_sim.envs.utils.info import Collision, CollisionOtherAgent, ReachGoal

PROGRESS_WEIGHT = 0.001

class SDOADRLDummyPolicy(object):
    """ the minimum viable version of a tensorflow-less SDOADRLDummyPolicy which is
    still compatible with soadrl env """
    name = 'SDOADRL'

    def configure(self, config):
        self.FOV_min_angle = config.getfloat(
            'map', 'angle_min') * np.pi % (2 * np.pi)
        self.FOV_max_angle = config.getfloat(
            'map', 'angle_max') * np.pi % (2 * np.pi)

    def human_state_in_FOV(self, self_state, human_state):
        rot = np.arctan2(
            human_state.py - self_state.py,
            human_state.px - self_state.px)
        angle = (rot - self_state.theta) % (2 * np.pi)
        return (angle > self.FOV_min_angle
                or angle < self.FOV_max_angle
                or self.FOV_min_angle == self.FOV_max_angle)

class NavRepTrainEnv(gym.Env):
    """ This class wraps the SOADRL env to make a RL ready simplified environment,
    compatible with navrep.

    Action space:
        (x velocity, y velocity, theta velocity), all in [m/s]
    Observation space:
        (scan, robotstate)
        scan:
            360 degree lidar scan, 1080 rays,
            with min angle of 0 (straight forwards), max angle of 2pi. [m]
        robotstate
            (goal x [m], goal y [m], vx [m/s], vy [m/s], vth [rad/s]) - all in robot frame
    """
    metadata = {'render.modes': ['human', 'rings']}

    def __init__(self, scenario='test', silent=False, legacy_mode=False, adaptive=True, lidar_legs=True,
                 collect_statistics=True):
        # gym env definition
        super(NavRepTrainEnv, self).__init__()
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Tuple((
            spaces.Box(low=-np.inf, high=np.inf, shape=(1080,), dtype=np.float32),
            spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        ))
        # parameters
        self.lidar_legs = lidar_legs
        self.silent = silent
        self.scenario = scenario
        self.adaptive = adaptive
        self.LEGACY_MODE = legacy_mode   # used for testing the original SOADRL policy
        self.collect_statistics = collect_statistics
        # other tools
        self.viewer = None
        self.episode_statistics = None
        if self.collect_statistics:
            self.episode_statistics = DataFrame(
                columns=[
                    "total_steps",
                    "scenario",
                    "damage",
                    "steps",
                    "goal_reached",
                    "reward",
                    "num_agents",
                    "num_walls",
                    "wall_time",
                ])
        self.total_steps = 0
        self.steps_since_reset = None
        self.episode_reward = None
        # conversion variables
        self.kLidarAngleIncrement = 0.00581718236208
        self.kLidarMergedMinAngle = 0
        self.kLidarMergedMaxAngle = 6.27543783188 + self.kLidarAngleIncrement
        self.n_angles = 1080
        self.lidar_scan = None
        self.lidar_angles = None
        self.border = None
        self.converter_cmap2d = CMap2D()
        self.converter_cmap2d.set_resolution(1.)
        self.distances_travelled_in_base_frame = None
        # environment
        self._make_env(silent=self.silent)

    def _convert_obs(self):
        robot = self.soadrl_sim.robot
        # lidar obs
        lidar_pos = np.array([robot.px, robot.py, robot.theta], dtype=np.float32)
        ranges = np.ones((self.n_angles,), dtype=np.float32) * 25.
        angles = np.linspace(self.kLidarMergedMinAngle,
                             self.kLidarMergedMaxAngle-self.kLidarAngleIncrement,
                             self.n_angles) + lidar_pos[2]
        render_contours_in_lidar(ranges, angles, self.flat_contours, lidar_pos[:2])
        # agents
        other_agents = []
        for i, human in enumerate(self.soadrl_sim.humans):
            pos = np.array([human.px, human.py, human.theta], dtype=np.float32)
            dist = self.distances_travelled_in_base_frame[i].astype(np.float32)
            vel = np.array([human.vx, human.vy], dtype=np.float32)
            if self.lidar_legs:
                agent = CSimAgent(pos, dist, vel)
            else:
                agent = CSimAgent(pos, dist, vel, type_="trunk", radius=human.radius)
            other_agents.append(agent)
        # apply through converter map (res 1., origin 0,0 -> i,j == x,y)
        self.converter_cmap2d.render_agents_in_lidar(ranges, angles, other_agents, lidar_pos[:2])
        self.lidar_scan = ranges
        self.lidar_angles = angles
        # robotstate obs
        # shape (n_agents, 5 [grx, gry, vx, vy, vtheta]) - all in base frame
        baselink_in_world = np.array([robot.px, robot.py, robot.theta])
        world_in_baselink = inverse_pose2d(baselink_in_world)
        robotvel_in_world = np.array([robot.vx, robot.vy, 0])  # TODO: actual robot rot vel?
        robotvel_in_baselink = apply_tf_to_vel(robotvel_in_world, world_in_baselink)
        goal_in_world = np.array([robot.gx, robot.gy, 0])
        goal_in_baselink = apply_tf_to_pose(goal_in_world, world_in_baselink)
        robotstate_obs = np.hstack([goal_in_baselink[:2], robotvel_in_baselink])
        obs = (self.lidar_scan, robotstate_obs)
        return obs

    def _update_dist_travelled(self):
        """ update dist travel var used for animating legs """
        # for each human, get vel in base frame
        for i, human in enumerate(self.soadrl_sim.humans):
            # dig up rotational velocity from past states log
            vrot = 0.
            if len(self.soadrl_sim.states) > 1:
                vrot = (self.soadrl_sim.states[-1][1][i].theta
                        - self.soadrl_sim.states[-2][1][i].theta) / self._get_dt()
            # transform world vel to base vel
            baselink_in_world = np.array([human.px, human.py, human.theta])
            world_in_baselink = inverse_pose2d(baselink_in_world)
            vel_in_world_frame = np.array([human.vx, human.vy, vrot])
            vel_in_baselink_frame = apply_tf_to_vel(vel_in_world_frame, world_in_baselink)
            self.distances_travelled_in_base_frame[i, :] += vel_in_baselink_frame * self._get_dt()

    def _add_border_obstacle(self):
        all_agents = [self.soadrl_sim.robot] + self.soadrl_sim.humans + self.soadrl_sim.other_robots
        all_vertices = np.array(self.soadrl_sim.obstacle_vertices).reshape((-1,2))
        x_agents = [a.px for a in all_agents]
        y_agents = [a.py for a in all_agents]
        x_goals = [a.gx for a in all_agents]
        y_goals = [a.gy for a in all_agents]
        x_vertices = list(all_vertices[:,0])
        y_vertices = list(all_vertices[:,1])
        BORDER_OFFSET = 1.
        x_min = min(x_agents + x_goals + x_vertices) - BORDER_OFFSET
        x_max = max(x_agents + x_goals + x_vertices) + BORDER_OFFSET
        y_min = min(y_agents + y_goals + y_vertices) - BORDER_OFFSET
        y_max = max(y_agents + y_goals + y_vertices) + BORDER_OFFSET
        # anticlockwise -> bounding obstacle
        border_vertices = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)][::-1]
        self.border = [(x_min, x_max), (y_min, y_max)]
        self.soadrl_sim.obstacle_vertices.append(border_vertices)

    def step(self, action):
        self.steps_since_reset += 1
        self.total_steps += 1
        if not self.LEGACY_MODE:
            # convert action
            # SOADRL - rotation is dtheta
            # IAN    - rotation is dtheta/dt
            dr = action[2] * self._get_dt()
            #  SOADRL - instant rot, then vel
            #  IAN    - vel, then rot
            vx_baselink = action[0] * np.cos(-dr) - action[1] * np.sin(-dr)
            vy_baselink = action[0] * np.sin(-dr) + action[1] * np.cos(-dr)
            soadrl_action = ActionXYRot(vx_baselink, vy_baselink, dr)
            # step
            _, _, reward, done, info = self.soadrl_sim.step(
                soadrl_action, compute_local_map=False, border=self.border)
            self._update_dist_travelled()
            # convert observations
            obs = self._convert_obs()
        else:
            state, local_map, reward, done, info = self.soadrl_sim.step(
                action, compute_local_map=True, border=self.border)
            self._update_dist_travelled()
            self._convert_obs()
            obs = (state, local_map)
            self.lidar_scan = local_map * self.soadrl_sim.angular_map_max_range
            self.lidar_angles = np.linspace(self.soadrl_sim.angular_map_min_angle,
                                            self.soadrl_sim.angular_map_max_angle,
                                            len(local_map)
                                            ) + self.soadrl_sim.robot.theta
        # add progress reward
        progress_reward = 0
        if not done:
            if len(self.soadrl_sim.states) > 1:
                new_s = self.soadrl_sim.states[-1][0]
                old_s = self.soadrl_sim.states[-2][0]
                new_p = np.array([new_s.px, new_s.py], dtype=np.double)
                old_p = np.array([old_s.px, old_s.py], dtype=np.double)
                goal = np.array([new_s.gx, new_s.gy], dtype=np.double)
                progress_reward = PROGRESS_WEIGHT*(fast_2f_norm(old_p - goal) - fast_2f_norm(new_p - goal))
        reward = reward + progress_reward
        reward = reward * 100.
        self.episode_reward += reward
        # adaptive difficulty
        if done and self.adaptive and self.scenario == "train":
            if isinstance(info, ReachGoal):
                self.soadrl_sim.human_num += 1
                self.soadrl_sim.num_walls += 1
                self.soadrl_sim.num_circles += 1
            else:
                self.soadrl_sim.human_num -= 1
                self.soadrl_sim.num_walls -= 1
                self.soadrl_sim.num_circles -= 1
            self.soadrl_sim.human_num = np.clip(self.soadrl_sim.human_num, 0, 5)
            self.soadrl_sim.num_walls = np.clip(self.soadrl_sim.num_walls, 0, 10)
            self.soadrl_sim.num_circles = np.clip(self.soadrl_sim.num_circles, 0, 10)
        # log data
        if done:
            if self.collect_statistics:
                self.episode_statistics.loc[len(self.episode_statistics)] = [
                    self.total_steps,
                    'navreptrain'+self.scenario,
                    100 if isinstance(info, Collision) or isinstance(info, CollisionOtherAgent) else 0,
                    self.steps_since_reset,
                    isinstance(info, ReachGoal),
                    reward,
                    self.soadrl_sim.human_num,
                    self.soadrl_sim.num_walls,
                    time.time(),
                ]
        return obs, reward, done, {'event': info}

    def reset(self):
        self.steps_since_reset = 0
        self.episode_reward = 0
        _, _ = self.soadrl_sim.reset(self.scenario, compute_local_map=False)
        random_rot = ActionXYRot(0, 0, 10.*(np.random.random()-0.5))
        self.soadrl_sim.step(random_rot, compute_local_map=False, border=self.border)
        if not self.LEGACY_MODE:
            self._add_border_obstacle()
        contours = self.soadrl_sim.obstacle_vertices
        self.flat_contours = flatten_contours(contours)
        self.distances_travelled_in_base_frame = np.zeros((len(self.soadrl_sim.humans), 3))
        obs = self._convert_obs()
        if self.LEGACY_MODE:
            state, local_map, reward, done, info = self.soadrl_sim.step(
                ActionXYRot(0, 0, 0), compute_local_map=True, border=self.border)
            obs = (state, local_map)
        return obs

    def render(self, mode='human', close=False,
               RENDER_LIDAR=True, lidar_scan_override=None, goal_override=None, save_to_file=False,
               show_score=False, robocentric=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
            return
        if self.lidar_scan is None:  # check that reset has been called
            return
        if mode == 'traj':
            self.soadrl_sim.render('traj')
        elif mode in ['human', 'rings']:
            # Window and viewport size
            WINDOW_W = 256
            WINDOW_H = 256
            VP_W = WINDOW_W
            VP_H = WINDOW_H
            from gym.envs.classic_control import rendering
            import pyglet
            from pyglet import gl
            # Create viewer
            if self.viewer is None:
                self.viewer = rendering.Viewer(WINDOW_W, WINDOW_H)
                self.transform = rendering.Transform()
                self.transform.set_scale(10, 10)
                self.transform.set_translation(128, 128)
                self.score_label = pyglet.text.Label(
                    '0000', font_size=12,
                    x=20, y=WINDOW_H*2.5/40.00, anchor_x='left', anchor_y='center',
                    color=(255,255,255,255))
#                 self.transform = rendering.Transform()
                self.currently_rendering_iteration = 0
                self.image_lock = threading.Lock()

            def make_circle(c, r, res=10):
                thetas = np.linspace(0, 2*np.pi, res+1)[:-1]
                verts = np.zeros((res, 2))
                verts[:,0] = c[0] + r * np.cos(thetas)
                verts[:,1] = c[1] + r * np.sin(thetas)
                return verts
            # Render in pyglet
            with self.image_lock:
                self.currently_rendering_iteration += 1
                self.viewer.draw_circle(r=10, color=(0.3,0.3,0.3))
                win = self.viewer.window
                win.switch_to()
                win.dispatch_events()
                win.clear()
                gl.glViewport(0, 0, VP_W, VP_H)
                # colors
                bgcolor = np.array([0.4, 0.8, 0.4])
                obstcolor = np.array([0.3, 0.3, 0.3])
                goalcolor = np.array([1., 1., 0.3])
                goallinecolor = 0.9 * bgcolor
                nosecolor = np.array([0.3, 0.3, 0.3])
                lidarcolor = np.array([1., 0., 0.])
                agentcolor = np.array([0., 1., 1.])
                # Green background
                gl.glBegin(gl.GL_QUADS)
                gl.glColor4f(bgcolor[0], bgcolor[1], bgcolor[2], 1.0)
                gl.glVertex3f(0, VP_H, 0)
                gl.glVertex3f(VP_W, VP_H, 0)
                gl.glVertex3f(VP_W, 0, 0)
                gl.glVertex3f(0, 0, 0)
                gl.glEnd()
                # Transform
                rx = self.soadrl_sim.robot.px
                ry = self.soadrl_sim.robot.py
                rth = self.soadrl_sim.robot.theta
                if robocentric:
                    # sets viewport = robocentric a.k.a T_sim_in_viewport = T_sim_in_robocentric
                    from pose2d import inverse_pose2d
                    T_sim_in_robot = inverse_pose2d(np.array([rx, ry, rth]))
                    # T_robot_in_robocentric is trans(128, 128), scale(10), rot(90deg)
                    # T_sim_in_robocentric = T_sim_in_robot * T_robot_in_robocentric
                    rot = np.pi / 2.
                    scale = 20
                    trans = (WINDOW_W / 2., WINDOW_H / 2.)
                    T_sim_in_robocentric = [
                        trans[0] + scale * (T_sim_in_robot[0] * np.cos(rot) - T_sim_in_robot[1] * np.sin(rot)),
                        trans[1] + scale * (T_sim_in_robot[0] * np.sin(rot) + T_sim_in_robot[1] * np.cos(rot)),
                        T_sim_in_robot[2] + rot,
                    ]
                    self.transform.set_translation(T_sim_in_robocentric[0], T_sim_in_robocentric[1])
                    self.transform.set_rotation(T_sim_in_robocentric[2])
                    self.transform.set_scale(scale, scale)
#                     self.transform.set_scale(20, 20)
                self.transform.enable()  # applies T_sim_in_viewport to below coords (all in sim frame)
                # Map closed obstacles ---
                for poly in self.soadrl_sim.obstacle_vertices:
                    gl.glBegin(gl.GL_LINE_LOOP)
                    gl.glColor4f(obstcolor[0], obstcolor[1], obstcolor[2], 1)
                    for vert in poly:
                        gl.glVertex3f(vert[0], vert[1], 0)
                    gl.glEnd()
                # LIDAR
                if RENDER_LIDAR:
                    px = self.soadrl_sim.robot.px
                    py = self.soadrl_sim.robot.py
                    angle = self.soadrl_sim.robot.theta
                    # LIDAR rays
                    scan = lidar_scan_override
                    if scan is None:
                        scan = self.lidar_scan
                    lidar_angles = self.lidar_angles
                    x_ray_ends = px + scan * np.cos(lidar_angles)
                    y_ray_ends = py + scan * np.sin(lidar_angles)
                    is_in_fov = np.cos(lidar_angles - angle) >= 0.78
                    for ray_idx in range(len(scan)):
                        end_x = x_ray_ends[ray_idx]
                        end_y = y_ray_ends[ray_idx]
                        gl.glBegin(gl.GL_LINE_LOOP)
                        if is_in_fov[ray_idx]:
                            gl.glColor4f(1., 1., 0., 0.1)
                        else:
                            gl.glColor4f(lidarcolor[0], lidarcolor[1], lidarcolor[2], 0.1)
                        gl.glVertex3f(px, py, 0)
                        gl.glVertex3f(end_x, end_y, 0)
                        gl.glEnd()
                # Agent body
                for n, agent in enumerate([self.soadrl_sim.robot] + self.soadrl_sim.humans):
                    px = agent.px
                    py = agent.py
                    angle = agent.theta
                    r = agent.radius
                    # Agent as Circle
                    poly = make_circle((px, py), r)
                    gl.glBegin(gl.GL_POLYGON)
                    if n == 0:
                        color = np.array([1., 1., 1.])
                    else:
                        color = agentcolor
                    gl.glColor4f(color[0], color[1], color[2], 1)
                    for vert in poly:
                        gl.glVertex3f(vert[0], vert[1], 0)
                    gl.glEnd()
                    # Direction triangle
                    xnose = px + r * np.cos(angle)
                    ynose = py + r * np.sin(angle)
                    xright = px + 0.3 * r * -np.sin(angle)
                    yright = py + 0.3 * r * np.cos(angle)
                    xleft = px - 0.3 * r * -np.sin(angle)
                    yleft = py - 0.3 * r * np.cos(angle)
                    gl.glBegin(gl.GL_TRIANGLES)
                    gl.glColor4f(nosecolor[0], nosecolor[1], nosecolor[2], 1)
                    gl.glVertex3f(xnose, ynose, 0)
                    gl.glVertex3f(xright, yright, 0)
                    gl.glVertex3f(xleft, yleft, 0)
                    gl.glEnd()
                # Goal
                xgoal = self.soadrl_sim.robot.gx
                ygoal = self.soadrl_sim.robot.gy
                r = self.soadrl_sim.robot.radius
                if goal_override is not None:
                    xgoal, ygoal = goal_override
                # Goal markers
                gl.glBegin(gl.GL_TRIANGLES)
                gl.glColor4f(goalcolor[0], goalcolor[1], goalcolor[2], 1)
                triangle = make_circle((xgoal, ygoal), r, res=3)
                for vert in triangle:
                    gl.glVertex3f(vert[0], vert[1], 0)
                gl.glEnd()
                # Goal line
                gl.glBegin(gl.GL_LINE_LOOP)
                gl.glColor4f(goallinecolor[0], goallinecolor[1], goallinecolor[2], 1)
                gl.glVertex3f(rx, ry, 0)
                gl.glVertex3f(xgoal, ygoal, 0)
                gl.glEnd()
                # --
                self.transform.disable()
                # Text
                self.score_label.text = ""
                if show_score:
                    self.score_label.text = "R {}".format(self.episode_reward)
                self.score_label.draw()
                win.flip()
                if save_to_file:
                    pyglet.image.get_buffer_manager().get_color_buffer().save(
                        "/tmp/navreptrainenv{:05}.png".format(self.total_steps))
                return self.viewer.isopen
        else:
            raise NotImplementedError

    def close(self):
        self.render(close=True)

    def _make_env(self, silent=False):
        # Create env
        config_dir = resource_filename('crowd_nav', 'config')
        config_file = os.path.join(config_dir, 'test_soadrl_static.config')
        config_file = os.path.expanduser(config_file)
        config = configparser.RawConfigParser()
        config.read(config_file)

        env = gym.make('CrowdSim-v0')
        env.configure(config, silent=silent)
        robot = Robot(config, 'humans')
        env.set_robot(robot)

        policy = SDOADRLDummyPolicy()
        policy.configure(config)
        if self.LEGACY_MODE:
            sess = tf.Session()
            policy = SDOADRL()
            policy.configure(sess, 'global', config)
            policy.set_phase('test')
            policy.load_model(os.path.expanduser('~/soadrl/Final_models/angular_map_full_FOV/rl_model'))

        env.robot.set_policy(policy)
        if not silent:
            env.robot.print_info()

        self.soadrl_sim = env

    def _get_viewer(self):
        return self.viewer

    def _get_dt(self):
        return self.soadrl_sim.time_step


if __name__ == "__main__":
    from navrep.tools.envplayer import EnvPlayer

    env = NavRepTrainEnv()
    player = EnvPlayer(env)
