import numpy as np
from gym import spaces

from navrep.tools.rings import generate_rings
from navrep.envs.navreptrainenv import NavRepTrainEnv
from navrep.envs.ianenv import IANEnv

_L = 1080  # lidar size
_RS = 5  # robotstate size
_64 = 64  # ring size

class FlatLidarAndStateEncoder(object):
    """ Generic class to encode the observations of an environment into a single 1d vector """
    def __init__(self):
        self._N = _L + _RS
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self._N,1), dtype=np.float32)

    def reset(self):
        pass

    def close(self):
        pass

    def _encode_obs(self, obs, action):
        lidar, state = obs
        e2e_obs = np.concatenate([lidar, state]).reshape(self._N,1)
        return e2e_obs

class RingsLidarAndStateEncoder(object):
    """ Generic class to encode the observations of an environment into a rings 2d image """
    def __init__(self):
        self._N = _64*_64 + _RS
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self._N,1), dtype=np.float32)
        self.rings_def = generate_rings(_64, _64)

    def reset(self):
        pass

    def close(self):
        pass

    def _encode_obs(self, obs, action):
        lidar, state = obs
        rings = (
            self.rings_def["lidar_to_rings"](lidar[None, :]).astype(float)
            / self.rings_def["rings_to_bool"]
        )
        e2e_obs = np.concatenate([rings.reshape(_64*_64), state]).reshape(self._N,1)
        return e2e_obs

class E2E1DNavRepEnv(NavRepTrainEnv):
    """ takes a (2) action as input
    outputs encoded obs (1085) """
    def __init__(self, *args, **kwargs):
        self.encoder = FlatLidarAndStateEncoder()
        super(E2E1DNavRepEnv, self).__init__(*args, **kwargs)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = self.encoder.observation_space

    def step(self, action):
        action = np.array([action[0], action[1], 0.])  # no rotation
        obs, reward, done, info = super(E2E1DNavRepEnv, self).step(action)
        h = self.encoder._encode_obs(obs, action)
        return h, reward, done, info

    def reset(self):
        self.encoder.reset()
        obs = super(E2E1DNavRepEnv, self).reset()
        h = self.encoder._encode_obs(obs, np.array([0,0,0]))
        return h

class E2ENavRepEnv(NavRepTrainEnv):
    """ takes a (2) action as input
    outputs encoded obs (1085) """
    def __init__(self, *args, **kwargs):
        self.encoder = RingsLidarAndStateEncoder()
        super(E2ENavRepEnv, self).__init__(*args, **kwargs)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = self.encoder.observation_space

    def step(self, action):
        action = np.array([action[0], action[1], 0.])  # no rotation
        obs, reward, done, info = super(E2ENavRepEnv, self).step(action)
        h = self.encoder._encode_obs(obs, action)
        return h, reward, done, info

    def reset(self):
        self.encoder.reset()
        obs = super(E2ENavRepEnv, self).reset()
        h = self.encoder._encode_obs(obs, np.array([0,0,0]))
        return h

class E2E1DIANEnv(IANEnv):
    """ takes a (2) action as input
    outputs encoded obs (1085) """
    def __init__(self, *args, **kwargs):
        self.encoder = FlatLidarAndStateEncoder()
        super(E2E1DIANEnv, self).__init__(*args, **kwargs)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = self.encoder.observation_space

    def step(self, action):
        action = np.array([action[0], action[1], 0.])  # no rotation
        obs, reward, done, info = super(E2E1DIANEnv, self).step(action)
        h = self.encoder._encode_obs(obs, action)
        return h, reward, done, info

    def reset(self):
        self.encoder.reset()
        obs = super(E2E1DIANEnv, self).reset()
        h = self.encoder._encode_obs(obs, np.array([0,0,0]))
        return h

class E2EIANEnv(IANEnv):
    """ takes a (2) action as input
    outputs encoded obs (1085) """
    def __init__(self, *args, **kwargs):
        self.encoder = RingsLidarAndStateEncoder()
        super(E2EIANEnv, self).__init__(*args, **kwargs)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = self.encoder.observation_space

    def step(self, action):
        action = np.array([action[0], action[1], 0.])  # no rotation
        obs, reward, done, info = super(E2EIANEnv, self).step(action)
        h = self.encoder._encode_obs(obs, action)
        return h, reward, done, info

    def reset(self):
        self.encoder.reset()
        obs = super(E2EIANEnv, self).reset()
        h = self.encoder._encode_obs(obs, np.array([0,0,0]))
        return h
