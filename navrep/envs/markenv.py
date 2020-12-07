from __future__ import print_function
import time
import numpy as np
from pepper_2d_simulator import parse_PepperRLEnv_args, PepperRLEnv, check_PepperRLEnv_args
import os
import gym
from gym import spaces
from pandas import DataFrame
from pkg_resources import resource_filename

FIRST_TRAIN_MAPS = ["marktrain"]
FIRST_TEST_MAPS = ["marksimple", "markcomplex"]

SECOND_TRAIN_MAPS = ["marksimple", "markcomplex", "marktm1", "marktm2", "marktm3"]
SECOND_TEST_MAPS = ["markclutter", "markmaze"]

ALL_MAPS = FIRST_TRAIN_MAPS + SECOND_TRAIN_MAPS + SECOND_TEST_MAPS

DEFAULT_MAPS = SECOND_TEST_MAPS

class MarkEnv(gym.Env):
    """ This class wraps the IARLenv to encode the observation,
    resulting in z + h encoding instead of raw lidar scans.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, silent=False, maps=DEFAULT_MAPS, max_episode_length=1000, collect_trajectories=True):
        # gym env definition
        super(MarkEnv, self).__init__()
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Tuple((
            spaces.Box(low=0, high=np.inf, shape=(1080,), dtype=np.float32),
            spaces.Box(low=0, high=np.inf, shape=(5,), dtype=np.float32),
        ))
        # constants
        self.COLLECT_TRAJECTORIES = collect_trajectories
        self.DT = 0.5  # should be the same as data rnn was trained with
        # parameters
        self.maps = maps
        self.silent = silent
        self.max_episode_length = max_episode_length
        self.max_episode_damage = 0
        # create underlying env
        self._make_env()
        # other tools
        self.episode_statistics = DataFrame(columns=[
            "total_steps",
            "scenario",
            "damage",
            "steps",
            "goal_reached",
            "timed_out",
            "collisioned_out",
            "goal",
            "trajectory",
            "reward",
            "wall_time",
        ])
        self.current_episode_trajectory = []
        self.current_episode_goal = None
        self.total_steps = 0
        # environment state variables
        self.reset()

    def step(self, action):
        self.steps_since_reset += 1
        self.total_steps += 1
        obs, reward, done, info = self.rlenv.step(action.reshape(1,3), ONLY_FOR_AGENT_0=True)
        timed_out = False
        collisioned_out = False
        if self.rlenv.episode_step[0] >= self.max_episode_length:
            done = True
            timed_out = True
        if self.rlenv.episode_damage[0] > 0:
            done = True
            collisioned_out = True
        if info["goal_reached"]:
            done = True
        # add to episodes log
        self.current_episode_trajectory.append(self.rlenv.virtual_peppers[0].pos * 1.)
        if done:
            traj = None
            if self.COLLECT_TRAJECTORIES:
                traj = np.array(self.current_episode_trajectory) * 1.
            self.episode_statistics.loc[len(self.episode_statistics)] = [
                self.total_steps,
                self.rlenv.args.map_name,
                self.rlenv.episode_damage[0] * 1.,
                self.steps_since_reset,
                info["goal_reached"],
                timed_out,
                collisioned_out,
                self.current_episode_goal * 1.,
                traj,
                reward,
                time.time(),
            ]
        obs = (obs[0][-1, :, 0], obs[1])  # latest scan only (buffer, ray, channel)
        return obs, reward, done, info

    def reset(self, set_scenario=None):
        if len(self.maps) > 1 or set_scenario is not None:
            self._make_env(set_scenario=set_scenario)
        obs = self.rlenv.reset(ONLY_FOR_AGENT_0=True)
        self.steps_since_reset = 0
        obs = (obs[0][-1, :, 0], obs[1])  # latest scan only (buffer, ray, channel)
        self.current_episode_trajectory = []
        self.current_episode_goal = self.rlenv.agent_goals[0] * 1.
        return obs

    def render(self, mode="human", close=False, lidar_scan_override=None, save_to_file=False):
        self.rlenv.render(mode=mode, close=close, lidar_scan_override=lidar_scan_override,
                          save_to_file=save_to_file)

    def close(self):
        try:
            self.rlenv.close()
        except AttributeError:
            pass

    def _get_viewer(self):
        viewer = None
        try:
            viewer = self.rlenv.viewer
        except AttributeError:
            pass
        return viewer

    def _make_env(self, set_scenario=None):
        try:
            self.rlenv.close()
        except AttributeError:
            pass
        args, _ = parse_PepperRLEnv_args(args=[])
        args.n_agents = 1
        map_name = set_scenario
        if map_name is None:
            map_name = np.random.choice(self.maps)
        args.map_name = map_name
        args.mode = "BOUNCE"
        args.continuous = True
        args.no_ros = True
        args.map_folder = resource_filename('asl_pepper_2d_sim_maps', 'maps')
        args.dt = self.DT
        check_PepperRLEnv_args(args)
        self.rlenv = PepperRLEnv(args, silent=self.silent)
        self.rlenv.virtual_peppers[0].NO_INERTIA = True

    def _get_dt(self):
        return self.rlenv.DT


if __name__ == "__main__":
    from navrep.tools.envplayer import EnvPlayer

    env = MarkEnv()
    player = EnvPlayer(env)
