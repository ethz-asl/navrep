from __future__ import print_function
import time
import gc
import numpy as np
from pepper_2d_iarlenv import parse_iaenv_args, IARLEnv, check_iaenv_args
import gym
from gym import spaces
from pandas import DataFrame

from navrep.envs.scenario_list import set_rl_scenario

PUNISH_SPIN = True

class IANEnv(gym.Env):
    """ This class wraps the IARLenv to make a RL ready simplified environment
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, silent=False, max_episode_length=1000, collect_trajectories=False):
        # gym env definition
        super(IANEnv, self).__init__()
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(1080,), dtype=np.float32)
        # parameters
        self.silent = silent
        self.max_episode_length = max_episode_length
        self.max_episode_damage = 5.  # leads to ~0.5% of the original reward
        # constants
        self.COLLECT_TRAJECTORIES = collect_trajectories
        self.TEMPERATURE = 0.5
        self.DT = 0.2  # should be the same as data rnn was trained with
        # other tools
        self.viewer = None
        self.episode_statistics = DataFrame(
            columns=[
                "total_steps",
                "scenario",
                "damage",
                "steps",
                "goal_reached",
                "timed_out",
                "collisioned_out",
                "static_damage_proportion",
                "goal",
                "trajectory",
                "other_trajectories",
                "reward",
                "wall_time",
            ])
        self.current_episode_trajectory = []
        self.current_episode_o_trajectories = []
        self.current_episode_goal = None
        self.total_steps = 0
        # environment state variables
        self.reset()

    def step(self, action):
        self.steps_since_reset += 1
        self.total_steps += 1
#         action = np.array([action[0], action[1], 0.])  # no rotation
        obs, reward, done, info = self.iarlenv.step(action, ONLY_FOR_AGENT_0=True)
        timed_out = self.iarlenv.rlenv.episode_step[0] >= self.max_episode_length
        collisioned_out = self.iarlenv.rlenv.episode_damage[0] >= self.max_episode_damage
        if self.iarlenv.rlenv.episode_damage[0] == 0:
            static_damage_proportion = np.nan
        else:
            static_damage_proportion = self.iarlenv.rlenv.episode_damage_from_static[0] / (
                self.iarlenv.rlenv.episode_damage[0])
        if info["goal_reached"]:
            done = True
        if timed_out:
            done = True
        if collisioned_out:
            done = True
        if PUNISH_SPIN:
            reward -= action[2]**2 / 1000.
        self.current_episode_trajectory.append(self.iarlenv.rlenv.virtual_peppers[0].pos * 1.)
        self.current_episode_o_trajectories.append(
            [vp.pos * 1. for vp in self.iarlenv.rlenv.virtual_peppers[1:]])
        if done:
            traj = None
            o_traj = None
            if self.COLLECT_TRAJECTORIES:
                traj = np.array(self.current_episode_trajectory) * 1.
                o_traj = np.array(self.current_episode_o_trajectories) * 1.
            self.episode_statistics.loc[len(self.episode_statistics)] = [
                self.total_steps,
                self.iarlenv.args.scenario,
                self.iarlenv.rlenv.episode_damage[0] * 1.,
                self.steps_since_reset,
                info["goal_reached"],
                timed_out,
                collisioned_out,
                static_damage_proportion,
                self.current_episode_goal * 1.,
                traj,
                o_traj,
                reward,
                time.time(),
            ]
        scan = obs[0]
        obs = (scan[-1][:, 0], obs[1])  # latest scan only (buffer, ray, channel)
        return obs, reward, done, info

    def reset(self, set_scenario=None):
        try:
            self.iarlenv.rlenv.viewer.close()
        except AttributeError:
            pass
        self._make_env(silent=self.silent, set_scenario=set_scenario)
        self.iarlenv.agents_pos0[0,2] = np.random.random() * 2 * np.pi
        obs = self.iarlenv.reset(ONLY_FOR_AGENT_0=True)
        self.steps_since_reset = 0
        self.current_episode_trajectory = []
        self.current_episode_o_trajectories = []
        self.current_episode_goal = self.iarlenv.rlenv.agent_goals[0] * 1.
        obs = (obs[0][-1,:,0], obs[1])
        return obs

    def render(self, *args, **kwargs):
        self.iarlenv.render(*args, **kwargs)

    def close(self):
        try:
            self.iarlenv.close()
        except AttributeError:
            pass

    def _make_env(self, silent=False, set_scenario=None):
        args = parse_iaenv_args(args=[])
        args.unmerged_scans = False
        args.continuous = True
        args.naive_plan = True
        args.no_ros = True
        args.no_pass_through = True
        set_rl_scenario(args, scenario_name=set_scenario)  # pick a random scenario
        check_iaenv_args(args)
        self.iarlenv = IARLEnv(args, silent=silent)
        self.iarlenv.rlenv.virtual_peppers[0].NO_INERTIA = True
        gc.collect()

    def _get_viewer(self):
        viewer = None
        try:
            viewer = self.iarlenv._get_viewer()
        except AttributeError:
            pass
        return viewer

    def _get_dt(self):
        return self.iarlenv.args.dt
