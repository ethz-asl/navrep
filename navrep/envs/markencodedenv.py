from gym import spaces
import numpy as np
import os

from navrep.envs.markenv import MarkEnv, DEFAULT_MAPS
from navrep.envs.encodedenv import EnvEncoder

class MarkOneEncoder(EnvEncoder):
    def __init__(self, backend, encoding,
                 gpu=False):
        assert backend == "VAE_LSTM"
        super(MarkOneEncoder, self).__init__(
            backend, encoding,
            rnn_model_path=os.path.expanduser("~/navrep/models/M/markonernn.json"),
            vae_model_path=os.path.expanduser("~/navrep/models/V/markonevae.json"),
            gpu=gpu,
        )

class MarkTwoEncoder(EnvEncoder):
    def __init__(self, backend, encoding,
                 gpu=False):
        assert backend == "VAE_LSTM"
        super(MarkTwoEncoder, self).__init__(
            backend, encoding,
            rnn_model_path=os.path.expanduser("~/navrep/models/M/marktwornn.json"),
            vae_model_path=os.path.expanduser("~/navrep/models/V/marktwovae.json"),
            gpu=gpu,
        )

class MarkOneEncodedEnv(MarkEnv):
    """ takes a (2) action as input
    outputs encoded obs (546) """
    def __init__(self, backend="VAE_LSTM", encoding="V_ONLY",
                 silent=False, maps=DEFAULT_MAPS, max_episode_length=1000, collect_trajectories=True):
        self.encoder = MarkOneEncoder(backend, encoding)
        super(MarkOneEncodedEnv, self).__init__(
            silent=silent, maps=maps, max_episode_length=max_episode_length,
            collect_trajectories=collect_trajectories)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = self.encoder.observation_space

    def step(self, action):
        action = np.array([action[0], action[1], 0.])  # no rotation
        obs, reward, done, info = super(MarkOneEncodedEnv, self).step(action)
        h = self.encoder._encode_obs(obs, action)
        return h, reward, done, info

    def reset(self, *args, **kwargs):
        self.encoder.reset()
        obs = super(MarkOneEncodedEnv, self).reset(*args, **kwargs)
        h = self.encoder._encode_obs(obs, np.array([0,0,0]))
        return h

    def render(self, mode="human", close=False):
        decoded_scan = self.encoder._get_last_decoded_scan()
        super(MarkOneEncodedEnv, self).render(mode=mode, close=close, lidar_scan_override=decoded_scan)

class MarkTwoEncodedEnv(MarkEnv):
    """ takes a (2) action as input
    outputs encoded obs (546) """
    def __init__(self, backend="VAE_LSTM", encoding="V_ONLY",
                 silent=False, maps=DEFAULT_MAPS, max_episode_length=1000, collect_trajectories=True):
        self.encoder = MarkTwoEncoder(backend, encoding)
        super(MarkTwoEncodedEnv, self).__init__(
            silent=silent, maps=maps, max_episode_length=max_episode_length,
            collect_trajectories=collect_trajectories)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = self.encoder.observation_space

    def step(self, action):
        action = np.array([action[0], action[1], 0.])  # no rotation
        obs, reward, done, info = super(MarkTwoEncodedEnv, self).step(action)
        h = self.encoder._encode_obs(obs, action)
        return h, reward, done, info

    def reset(self, *args, **kwargs):
        self.encoder.reset()
        obs = super(MarkTwoEncodedEnv, self).reset(*args, **kwargs)
        h = self.encoder._encode_obs(obs, np.array([0,0,0]))
        return h

    def render(self, mode="human", close=False):
        decoded_scan = self.encoder._get_last_decoded_scan()
        super(MarkTwoEncodedEnv, self).render(mode=mode, close=close, lidar_scan_override=decoded_scan)
