import numpy as np
import os
import configparser
import tensorflow as tf
from pkg_resources import resource_filename
from pyniel.python_tools.path_tools import make_dir_if_not_exists

import crowd_sim  # adds CrowdSim-v0 to gym  # noqa
from crowd_sim.envs.crowd_sim import CrowdSim  # reference to env code  # noqa
from crowd_sim.envs.utils.robot import Robot  # next line fails otherwise # noqa
from crowd_nav.policy.network_om import SDOADRL
from crowd_sim.envs.utils.state import JointState, FullState, ObservableState
from crowd_sim.envs.utils.action import ActionRot

from navrep.scripts.cross_test_navreptrain_in_ianenv import run_test_episodes
from navrep.tools.commonargs import parse_common_args
from navrep.envs.ianenv import IANEnv

TODO = None

class LuciaRawPolicy(object):
    """ legacy SOADRL policy from lucia's paper, takes in agents state, local map
    The problem is that in the original implementation, policy and environment are intertwined.
    this class goes further into separating them by reimplementing methods from
    agents.py, robots.py """

    def __init__(self):
        self._make_policy()

    def _make_policy(self):
        # Config
        config_dir = resource_filename('crowd_nav', 'config')
        config_file = os.path.join(config_dir, 'test_soadrl_static.config')
        config = configparser.RawConfigParser()
        config.read(config_file)

        sess = tf.Session()
        policy = SDOADRL()
        policy.configure(sess, 'global', config)
        policy.set_phase('test')
        self.model_path = os.path.expanduser('~/soadrl/Final_models/angular_map_full_FOV/rl_model')
        policy.load_model(self.model_path)

        self.policy = policy

    def act(self, obs):
        robot_state, humans_state, local_map = obs
        state = JointState(robot_state, humans_state)
        action = self.policy.predict(state, local_map, None)
        action = ActionRot(robot_state.v_pref * action.v, action.r)  # de-normalize
        return action

class IANEnvWithLegacySOADRLObs(object):
    def __init__(self,
                 silent=False, max_episode_length=1000, collect_trajectories=False):
        # Get lidar values from the SOADRL config
        config_dir = resource_filename('crowd_nav', 'config')
        config_file = os.path.join(config_dir, 'test_soadrl_static.config')
        config = configparser.RawConfigParser()
        config.read(config_file)
        self.v_pref = config.getfloat('humans', 'v_pref')
        # lidar scan expected by SOADRL
        self.angular_map_max_range = config.getfloat('map', 'angular_map_max_range')
        self.angular_map_dim = config.getint('map', 'angular_map_dim')
        self.angular_map_min_angle = config.getfloat('map', 'angle_min') * np.pi
        self.angular_map_max_angle = config.getfloat('map', 'angle_max') * np.pi
        self.angular_map_angle_increment = (
            self.angular_map_max_angle - self.angular_map_min_angle) / self.angular_map_dim
        self.lidar_upsampling = 15

        # create env
        self.env = IANEnv(
            silent=silent, max_episode_length=max_episode_length, collect_trajectories=collect_trajectories)
        self.reset()

    def reset(self):
        """ IANEnv destroys and re-creates its iarlenv at every reset, so apply our changes here """
        self.env.reset()

        # we raytrace at a higher resolution, then downsample back to the original soadrl resolution
        # this avoids missing small obstacles due to the small soadrl resolution
        self.env.iarlenv.rlenv.virtual_peppers[0].kLidarMergedMaxAngle = self.angular_map_max_angle
        self.env.iarlenv.rlenv.virtual_peppers[0].kLidarMergedMinAngle = self.angular_map_min_angle
        self.env.iarlenv.rlenv.virtual_peppers[0].kLidarAngleIncrement = \
            self.angular_map_angle_increment / self.lidar_upsampling
        self.env.iarlenv.rlenv.kMergedScanSize = self.angular_map_dim * self.lidar_upsampling

        self.episode_statistics = self.env.episode_statistics

        obs, _, _, _ = self.step(ActionRot(0.,0.))
        return obs

    def step(self, action):
        # convert lucia action to IANEnv action
        ianenv_action = np.array([0., 0., 0.])
        # SOADRL - rotation is dtheta
        # IAN    - rotation is dtheta/dt
        ianenv_action[2] = action.r / self.env._get_dt()
        #  SOADRL - instant rot, then vel
        #  IAN    - vel, then rot
        action_vy = 0. # SOADRL outputs non-holonomic by default
        ianenv_action[0] = action.v * np.cos(action.r) - action_vy * np.sin(action.r)
        ianenv_action[1] = action.v * np.sin(action.r) + action_vy * np.cos(action.r)
        # get obs from IANEnv
        obs, rew, done, info = self.env.step(ianenv_action)
        # convert to SOADRL style
        robot_state = FullState(
            self.env.iarlenv.rlenv.virtual_peppers[0].pos[0],
            self.env.iarlenv.rlenv.virtual_peppers[0].pos[1],
            self.env.iarlenv.rlenv.virtual_peppers[0].vel[0],
            self.env.iarlenv.rlenv.virtual_peppers[0].vel[1],
            self.env.iarlenv.rlenv.vp_radii[0],
            self.env.iarlenv.rlenv.agent_goals[0][0],
            self.env.iarlenv.rlenv.agent_goals[0][1],
            self.v_pref,
            self.env.iarlenv.rlenv.virtual_peppers[0].pos[2],)
        humans_state = [ObservableState(
            human.pos[0],
            human.pos[1],
            human.vel[0],
            human.vel[1],
            r,) for human, r in zip(
                self.env.iarlenv.rlenv.virtual_peppers[1:], self.env.iarlenv.rlenv.vp_radii[1:])]
        scan = obs[0]
        # for each angular section we take the min of the returns
        downsampled_scan = scan.reshape((-1, self.lidar_upsampling))
        downsampled_scan = np.min(downsampled_scan, axis=1)
        self.last_downsampled_scan = downsampled_scan
        local_map = np.clip(downsampled_scan / self.angular_map_max_range, 0., 1.)
        obs = (robot_state, humans_state, local_map)
        return obs, rew, done, info

    def _get_dt(self):
        return self.env._get_dt()

    def render(self, *args, **kwargs):
        _, lidar_angles = self.env.iarlenv.rlenv.virtual_peppers[0].get_lidar_update_ijangles(
            "merged", self.env.iarlenv.rlenv.kMergedScanSize
        )
        lidar_angles_downsampled = lidar_angles[::self.lidar_upsampling]
        kwargs["lidar_angles_override"] = lidar_angles_downsampled
        kwargs["lidar_scan_override"] = self.last_downsampled_scan
        return self.env.render(*args, **kwargs)


if __name__ == '__main__':
    args, _ = parse_common_args()

    if args.n is None:
        args.n = 1000
    collect_trajectories = False

    env = IANEnvWithLegacySOADRLObs(silent=True, collect_trajectories=collect_trajectories)
    policy = LuciaRawPolicy()

    S = run_test_episodes(env, policy, render=args.render, num_episodes=args.n)

    DIR = os.path.expanduser("~/navrep/eval/crosstest")
    if args.dry_run:
        DIR = "/tmp/navrep/eval/crosstest"
    make_dir_if_not_exists(DIR)
    if collect_trajectories:
        NAME = "lucianavreptrain_in_ianenv_{}.pckl".format(len(S))
        PATH = os.path.join(DIR, NAME)
        S.to_pickle(PATH)
    else:
        NAME = "lucianavreptrain_in_ianenv_{}.csv".format(len(S))
        PATH = os.path.join(DIR, NAME)
        S.to_csv(PATH)
    print("{} written.".format(PATH))
