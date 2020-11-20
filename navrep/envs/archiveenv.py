from __future__ import print_function
import os
import threading
import numpy as np
import gym
from gym import spaces

from navrep.tools.wdataset import WorldModelDataset
from navrep.envs.encodedenv import EnvEncoder

MAX_LIDAR_DIST = 25.  # used here for rendering purposes only

class ArchiveEnv(gym.Env):
    """ This class wraps the IARLenv to make a RL ready simplified environment
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, directory, file_limit=None, silent=False, max_episode_length=1000):
        # gym env definition
        super(ArchiveEnv, self).__init__()
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(1080,), dtype=np.float32)
        self.data = WorldModelDataset._load_data(None, directory, file_limit=file_limit)
        if len(self.data["scans"]) == 0:
            raise ValueError
        self.current_iteration = None
        self.data["dones"][-1] = 1
        self.viewer = None

    def _step(self, action):
        """ preserved when inherited classes overrite step() """
        scan = self.data["scans"][self.current_iteration].astype(np.float32)
        robotstate = self.data["robotstates"][self.current_iteration]
        reward = self.data["rewards"][self.current_iteration]
        done = self.data["dones"][self.current_iteration]
        obs = (scan, robotstate)  # latest scan only (buffer, ray, channel)
        if not done:
            self.current_iteration += 1
        return obs, reward, done, {}

    def step(self, action):
        return self._step(action)

    def reset(self):
        action = self.data["actions"][self.current_iteration]
        if self.current_iteration is None:
            self.current_iteration = 0
            obs, _, _, _ = self._step(action)
            return obs
        # skip to the start of next episode
        while True:
            obs, _, done, _ = self._step(action)
            if done:
                self.current_iteration += 1
                if self.current_iteration >= len(self.data["scans"]):
                    self.current_iteration = 0
                break
        return obs

    def close(self):
        self.render(close=True)

    def _get_viewer(self):
        return self.viewer

    def _get_dt(self):
        return 0.25

    def render(self, mode="human", close=False, lidar_scan_override=None, save_to_file=False,
               draw_score=True):
        if close:
            if self.viewer is not None:
                self.viewer.close()
            return

        # get last obs
        goal_pred = self.data["robotstates"][self.current_iteration-1][:2]
        action = self.data["actions"][self.current_iteration-1]
        scan = self.data["scans"][self.current_iteration-1]
        scan[scan == 0] = MAX_LIDAR_DIST  # render 0 returns as inf
        if lidar_scan_override is not None:
            scan = lidar_scan_override

        if mode == "rgb_array":
            raise NotImplementedError
        elif mode in ["human", "rings"]:
            # Window and viewport size
            WINDOW_W = 256
            WINDOW_H = 256
            M_PER_PX = 13.6 / WINDOW_H
            VP_W = WINDOW_W
            VP_H = WINDOW_H
            from gym.envs.classic_control import rendering
            import pyglet
            from pyglet import gl

            # enable this to render for hg_archivenv visual
            if False:
                draw_score = False
                save_to_file = True
                WINDOW_W = 512
                WINDOW_H = 512
                M_PER_PX = 51.2 / WINDOW_H

            # Create viewer
            if self.viewer is None:
                self.viewer = rendering.Viewer(WINDOW_W, WINDOW_H)
                self.score_label = pyglet.text.Label(
                    "0000",
                    font_size=12,
                    x=20,
                    y=WINDOW_H * 2.5 / 40.00,
                    anchor_x="left",
                    anchor_y="center",
                    color=(255, 255, 255, 255),
                )
                #                 self.transform = rendering.Transform()
                self.currently_rendering_iteration = 0
                self.image_lock = threading.Lock()
            # Render in pyglet
            def make_circle(c, r, res=10):
                thetas = np.linspace(0, 2 * np.pi, res + 1)[:-1]
                verts = np.zeros((res, 2))
                verts[:, 0] = c[0] + r * np.cos(thetas)
                verts[:, 1] = c[1] + r * np.sin(thetas)
                return verts

            with self.image_lock:
                self.currently_rendering_iteration += 1
                self.viewer.draw_circle(r=10, color=(0.3, 0.3, 0.3))
                win = self.viewer.window
                win.switch_to()
                win.dispatch_events()
                win.clear()
                gl.glViewport(0, 0, VP_W, VP_H)
                # colors
#                 bgcolor = np.array([0.4, 0.8, 0.4])
                bgcolor = np.array([0.7, 0.75, 0.86])
                nosecolor = np.array([0.3, 0.3, 0.3])
#                 lidarcolor = np.array([1.0, 0.0, 0.0])
                lidarcolor = np.array([1., 0.3, 0.0])
#                 fovcolor = np.array([1., 1., 0.])
                fovcolor = np.array([0.8, 0.8, 0.2])
                # Green background
                gl.glBegin(gl.GL_QUADS)
                gl.glColor4f(bgcolor[0], bgcolor[1], bgcolor[2], 1.0)
                gl.glVertex3f(0, VP_H, 0)
                gl.glVertex3f(VP_W, VP_H, 0)
                gl.glVertex3f(VP_W, 0, 0)
                gl.glVertex3f(0, 0, 0)
                gl.glEnd()
                # LIDAR
                i = WINDOW_W / 2.0
                j = WINDOW_H / 2.0
                angle = np.pi / 2.0
                lidar_angles = np.linspace(0, 2 * np.pi, len(scan) + 1)[:-1]
                lidar_angles = lidar_angles + np.pi / 2.  # make robot face up
                i_ray_ends = i + scan / M_PER_PX * np.cos(lidar_angles)
                j_ray_ends = j + scan / M_PER_PX * np.sin(lidar_angles)
                is_in_fov = np.cos(lidar_angles - angle) >= 0.78
                for ray_idx in range(len(scan)):
                    end_i = i_ray_ends[ray_idx]
                    end_j = j_ray_ends[ray_idx]
                    gl.glBegin(gl.GL_LINE_LOOP)
                    if is_in_fov[ray_idx]:
                        gl.glColor4f(fovcolor[0], fovcolor[1], fovcolor[2], 0.1)
                    else:
                        gl.glColor4f(lidarcolor[0], lidarcolor[1], lidarcolor[2], 0.1)
                    gl.glVertex3f(i, j, 0)
                    gl.glVertex3f(end_i, end_j, 0)
                    gl.glEnd()
                # Agent body
                i = WINDOW_W / 2.0
                j = WINDOW_H / 2.0
                r = 0.3 / M_PER_PX
                angle = np.pi / 2.0
                poly = make_circle((i, j), r)
                gl.glBegin(gl.GL_POLYGON)
                color = np.array([1.0, 1.0, 1.0])
                gl.glColor4f(color[0], color[1], color[2], 1)
                for vert in poly:
                    gl.glVertex3f(vert[0], vert[1], 0)
                gl.glEnd()
                # Direction triangle
                inose = i + r * np.cos(angle)
                jnose = j + r * np.sin(angle)
                iright = i + 0.3 * r * -np.sin(angle)
                jright = j + 0.3 * r * np.cos(angle)
                ileft = i - 0.3 * r * -np.sin(angle)
                jleft = j - 0.3 * r * np.cos(angle)
                gl.glBegin(gl.GL_TRIANGLES)
                gl.glColor4f(nosecolor[0], nosecolor[1], nosecolor[2], 1)
                gl.glVertex3f(inose, jnose, 0)
                gl.glVertex3f(iright, jright, 0)
                gl.glVertex3f(ileft, jleft, 0)
                gl.glEnd()
                # Goal
                goalcolor = np.array([1., 1., 0.3])
                px_goal = goal_pred / M_PER_PX
                igoal = i - px_goal[1]  # rotate 90deg to face up
                jgoal = j + px_goal[0]
                # Goal line
                gl.glBegin(gl.GL_LINE_LOOP)
                gl.glColor4f(goalcolor[0], goalcolor[1], goalcolor[2], 1)
                gl.glVertex3f(i, j, 0)
                gl.glVertex3f(igoal, jgoal, 0)
                gl.glEnd()
                # Goal markers
                gl.glBegin(gl.GL_TRIANGLES)
                gl.glColor4f(goalcolor[0], goalcolor[1], goalcolor[2], 1)
                triangle = make_circle((igoal, jgoal), r, res=3)
                for vert in triangle:
                    gl.glVertex3f(vert[0], vert[1], 0)
                gl.glEnd()
                # Text
                self.score_label.text = "File {} It {} a {:.1f} {:.1f} {:.1f}".format(
                    self.current_iteration // 1000,
                    self.current_iteration % 1000,
                    action[0],
                    action[1],
                    action[2],
                )
                if draw_score:
                    self.score_label.draw()
                win.flip()
                if save_to_file:
                    pyglet.image.get_buffer_manager().get_color_buffer().save(
                        "/tmp/archivenv{:05}.png".format(self.currently_rendering_iteration))
                return self.viewer.isopen


class ArchiveEncoder(EnvEncoder):
    def __init__(self, backend, encoding):
        assert backend == "VAE_LSTM"
        super(ArchiveEncoder, self).__init__(
            backend, encoding,
            rnn_model_path=os.path.expanduser("~/navrep/models/M/navreptrainrnn.json"),
            vae_model_path=os.path.expanduser("~/navrep/models/V/navreptrainvae.json"),
        )


class ArchiveEncodedEnv(ArchiveEnv):
    """ takes a (2) action as input
    outputs encoded obs (546) """
    def __init__(self, backend, encoding,
                 directory, file_limit=None, silent=False, max_episode_length=1000):
        self.encoder = ArchiveEncoder(backend, encoding)
        super(ArchiveEncodedEnv, self).__init__(directory, file_limit=file_limit, silent=silent,
                                                max_episode_length=max_episode_length)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = self.encoder.observation_space

    def step(self, action):
        action = np.array([action[0], action[1], 0.])  # no rotation
        obs, reward, done, info = super(ArchiveEncodedEnv, self).step(action)
        h = self.encoder._encode_obs(obs, action)
        return h, reward, done, info

    def reset(self, *args, **kwargs):
        self.encoder.reset()
        obs = super(ArchiveEncodedEnv, self).reset(*args, **kwargs)
        h = self.encoder._encode_obs(obs, np.array([0,0,0]))
        return h

    def close(self):
        super(ArchiveEncodedEnv, self).close()
        self.encoder.close()

    def render(self, mode="human", close=False, save_to_file=False):
        decoded_scan = self.encoder._get_last_decoded_scan()
        super(ArchiveEncodedEnv, self).render(
            mode=mode, close=close, lidar_scan_override=decoded_scan, save_to_file=save_to_file)
        if mode == "rings":
            self.encoder._render_rings(close=close, save_to_file=save_to_file)
