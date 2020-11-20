from gym import spaces
import numpy as np
import os

from navrep.envs.navreptrainenv import NavRepTrainEnv
from navrep.envs.encodedenv import EnvEncoder

class NavRepTrainEncoder(EnvEncoder):
    def __init__(self, backend, encoding,
                 gpu=False, encoder_to_share_model_with=None):
        super(NavRepTrainEncoder, self).__init__(
            backend, encoding,
            rnn_model_path=os.path.expanduser("~/navrep/models/M/navreptrainrnn.json"),
            rnn1d_model_path=os.path.expanduser("~/navrep/models/M/navreptrainrnn1d.json"),
            vae_model_path=os.path.expanduser("~/navrep/models/V/navreptrainvae.json"),
            vae1d_model_path=os.path.expanduser("~/navrep/models/V/navreptrainvae1d.json"),
            gpt_model_path=os.path.expanduser("~/navrep/models/W/navreptraingpt"),
            gpt1d_model_path=os.path.expanduser("~/navrep/models/W/navreptraingpt1d"),
            vae1dlstm_model_path=os.path.expanduser("~/navrep/models/W/navreptrainvae1dlstm"),
            vaelstm_model_path=os.path.expanduser("~/navrep/models/W/navreptrainvaelstm"),
            gpu=gpu,
            encoder_to_share_model_with=None,
        )


class NavRepTrainEncodedEnv(NavRepTrainEnv):
    """ takes a (2) action as input
    outputs encoded obs (546) """
    def __init__(self, backend, encoding,
                 scenario='test', silent=False, adaptive=True,
                 gpu=False, shared_encoder=None, encoder=None):
        if encoder is None:
            encoder = NavRepTrainEncoder(backend, encoding,
                                         gpu=gpu, encoder_to_share_model_with=shared_encoder)
        self.encoder = encoder
        super(NavRepTrainEncodedEnv, self).__init__(scenario=scenario, silent=silent, adaptive=adaptive,
                                                    legacy_mode=False)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = self.encoder.observation_space

    def step(self, action):
        action = np.array([action[0], action[1], 0.])  # no rotation
        obs, reward, done, info = super(NavRepTrainEncodedEnv, self).step(action)
        h = self.encoder._encode_obs(obs, action)
        return h, reward, done, info

    def reset(self, *args, **kwargs):
        self.encoder.reset()
        obs = super(NavRepTrainEncodedEnv, self).reset(*args, **kwargs)
        h = self.encoder._encode_obs(obs, np.array([0,0,0]))
        return h

    def close(self):
        super(NavRepTrainEncodedEnv, self).close()
        self.encoder.close()

    def render(self, mode="human", close=False, save_to_file=False,
               robocentric=False, render_decoded_scan=True):
        decoded_scan = None
        if render_decoded_scan:
            decoded_scan = self.encoder._get_last_decoded_scan()
        super(NavRepTrainEncodedEnv, self).render(
            mode=mode, close=close, lidar_scan_override=decoded_scan, save_to_file=save_to_file,
            robocentric=robocentric)
        if mode == "rings":
            self.encoder._render_rings(close=close, save_to_file=save_to_file)
        if mode == "polar":
            self.encoder._render_rings_polar(close=close, save_to_file=save_to_file)
