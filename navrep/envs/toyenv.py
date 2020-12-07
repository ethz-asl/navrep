import gym
import os
from gym import spaces
import numpy as np
from pepper_2d_iarlenv import parse_iaenv_args, IARLEnv, check_iaenv_args
from pkg_resources import resource_filename

class ToyEnv(gym.Env):
    def __init__(self):
        super(ToyEnv, self).__init__()
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        # actually obs space is Tuple(Box(1080), Box(5)) but no support from sb
        self.observation_space = spaces.Tuple((
            spaces.Box(low=0, high=np.inf, shape=(1080,), dtype=np.float32),
            spaces.Box(low=0, high=np.inf, shape=(5,), dtype=np.float32),
        ))
        # args
        args = parse_iaenv_args()
        args.unmerged_scans = False
        args.continuous = True
        args.naive_plan = True
        args.no_ros = True
        args.dt = 0.5
        args.map_folder = resource_filename('asl_pepper_2d_sim_maps', 'maps')
        args.scenario = "toyempty1"
        args.map_name = "empty"
        args.map_downsampling_passes = 0
        check_iaenv_args(args)
        # env
        self.iarlenv = IARLEnv(args)
        for env in self.iarlenv.rlenv.virtual_peppers:
            env.NO_INERTIA = True

    def step(self, action):
        obs, rew, done, info = self.iarlenv.step(action, ONLY_FOR_AGENT_0=True)
        if self.iarlenv.rlenv.episode_damage > 0:
            done = True
        obs = (obs[0][-1, :, 0], obs[1])  # latest scan only (buffer, ray, channel)
        return obs, rew, done, info

    def render(self, *args, **kwargs):
        self.iarlenv.render(*args, **kwargs)

    def reset(self):
        self.iarlenv.reset()

    def close(self):
        self.iarlenv.close()

    def _get_dt(self):
        return self.iarlenv.args.dt

    def _get_viewer(self):
        viewer = self.iarlenv._get_viewer()
        return viewer


if __name__ == "__main__":
    from navrep.tools.envplayer import EnvPlayer

    env = ToyEnv()
    player = EnvPlayer(env)
