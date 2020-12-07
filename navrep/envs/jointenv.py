from __future__ import print_function
import os
import time
import gc
import numpy as np
from pepper_2d_simulator import parse_PepperRLEnv_args, PepperRLEnv, check_PepperRLEnv_args
import gym
from gym import spaces
from pandas import DataFrame
from pkg_resources import resource_filename

from navrep.envs.scenario_list import map_downsampling

PUNISH_SPIN = False

# TODO: this is still a work in progress. @Daniel: Make it into a VecEnv
# from stable_baselines import DummyVecEnv

CROWDMOVE_MAPS = ["crowdmove1", "crowdmove2", "crowdmove3", "crowdmove4", "crowdmove5", "crowdmove6"]

class JointEnv(gym.Env):
    """ This class wraps the IARLenv to make a RL ready simplified environment
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, silent=False, n_robots=10, maps=CROWDMOVE_MAPS, circular_scenario=False,
                 max_episode_length=1000, no_inertia=True, collect_trajectories=False, lidar_legs=False):
        # gym env definition
        super(JointEnv, self).__init__()
        self.action_space = spaces.Box(low=-1, high=1, shape=(n_robots, 3), dtype=np.float32)
        self.observation_space = spaces.Tuple((spaces.Box(low=-np.inf, high=np.inf,
                                                          shape=(n_robots, 4, 1080, 1), dtype=np.float32),
                                              spaces.Box(low=-np.inf, high=np.inf,
                                                         shape=(n_robots, 5), dtype=np.float32)))
        # parameters
        self.silent = silent
        self.n_robots = n_robots
        self.maps = maps
        self.no_inertia = no_inertia
        self.circular_scenario = circular_scenario
        self.lidar_legs = lidar_legs
        # constants
        self.COLLECT_TRAJECTORIES = collect_trajectories
        self.TEMPERATURE = 0.5
        self.DT = 0.2  # should be the same as data rnn was trained with
        # other tools
        self.viewer = None
        self.episode_statistics = DataFrame(
            columns=[
                "total_steps",
                "mode",
                "steps",
                "goal_reached",
                "damage",
                "static_damage",
                "goal",
                "trajectory",
                "other_trajectories",
                "reward",
                "wall_time",
            ])
        self.current_episode_trajectory = []
        self.current_episode_o_trajectories = []
        self.current_episode_goals = None
        self.total_steps = 0
        # environment state variables
        self.reset()

    def step(self, action):
        if action.shape == (3,):
            action = np.repeat(action[None, :], self.n_robots, axis=0)
        self.steps_since_reset += 1
        self.total_steps += 1
        obs, reward, done, infos = self.rlenv.step(action)
        if PUNISH_SPIN:
            reward -= action[2]**2 / 1000.
        self.current_episode_trajectory.append(self.rlenv.virtual_peppers[0].pos * 1.)
        self.current_episode_o_trajectories.append(
            [vp.pos * 1. for vp in self.rlenv.virtual_peppers[1:]])
        if np.any(done):
            traj = None
            o_traj = None
            if self.COLLECT_TRAJECTORIES:
                traj = np.array(self.current_episode_trajectory) * 1.
                o_traj = np.array(self.current_episode_o_trajectories) * 1.
            self.episode_statistics.loc[len(self.episode_statistics)] = [
                self.total_steps,
                self.rlenv.MODE,
                self.steps_since_reset,
                [info["goal_reached"] for info in infos],
                self.rlenv.episode_damage * 1.,
                self.rlenv.episode_damage_from_static * 1.,
                self.current_episode_goals * 1.,
                traj,
                o_traj,
                reward,
                time.time(),
            ]
        return obs, reward, np.any(done), infos

    def reset(self, set_scenario=None):
        try:
            self.rlenv.viewer.close()
        except AttributeError:
            pass
        self._make_env(silent=self.silent, set_scenario=set_scenario)
        obs = self.rlenv.reset()
        self.steps_since_reset = 0
        self.current_episode_trajectory = []
        self.current_episode_o_trajectories = []
        self.current_episode_goals = self.rlenv.agent_goals * 1.
        return obs

    def render(self, *args, **kwargs):
        self.rlenv.render(*args, **kwargs)

    def close(self):
        try:
            self.rlenv.close()
        except AttributeError:
            pass

    def _make_env(self, silent=False, set_scenario=None):
        args, _ = parse_PepperRLEnv_args(args=[])
        args.unmerged_scans = False
        args.continuous = True
        args.no_ros = True
        args.no_legs = not self.lidar_legs
        args.n_agents = self.n_robots
        args.map_folder = resource_filename('asl_pepper_2d_sim_maps', 'maps')
        args.map_name = set_scenario
        if args.map_name is None:
            args.map_name = np.random.choice(self.maps)
        args.map_downsampling_passes = map_downsampling[args.map_name]
        # only do circular scenario in empty map
        args.circular_scenario = False
        if self.circular_scenario and args.map_name == "crowdmove1":
            args.circular_scenario = True
            args.map_downsampling_passes = 0
        check_PepperRLEnv_args(args)
        self.rlenv = PepperRLEnv(args, silent=silent)
        self.rlenv.virtual_peppers[0].NO_INERTIA = self.no_inertia
        gc.collect()

    def _get_viewer(self):
        viewer = None
        try:
            viewer = self.rlenv._get_viewer()
        except AttributeError:
            pass
        return viewer

    def _get_dt(self):
        return self.rlenv.args.dt


if __name__ == "__main__":
    from navrep.tools.envplayer import EnvPlayer
    env = JointEnv()
    player = EnvPlayer(env)
