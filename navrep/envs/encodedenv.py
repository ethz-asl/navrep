from __future__ import print_function
import numpy as np
import os
from gym import spaces

from navrep.tools.rings import generate_rings
from navrep.envs.ianenv import IANEnv
from navrep.models.rnn import (reset_graph, sample_hps_params, MDNRNN,
                               rnn_init_state, rnn_next_state, MAX_GOAL_DIST)
from navrep.models.vae2d import ConvVAE
from navrep.models.vae1d import Conv1DVAE
from navrep.models.gpt import GPT, GPTConfig, load_checkpoint
from navrep.models.gpt1d import GPT1D
from navrep.models.vae1dlstm import VAE1DLSTM, VAE1DLSTMConfig
from navrep.models.vaelstm import VAELSTM, VAELSTMConfig
from navrep.tools.wdataset import scans_to_lidar_obs

PUNISH_SPIN = True

""" VM backends: VAE_LSTM, W backends: GPT, GPT1D, VAE1DLSTM """
""" ENCODINGS: V_ONLY, VM, M_ONLY """
_G = 2  # goal dimensions
_A = 3  # action dimensions
_RS = 5  # robot state
_64 = 64  # ring size
_L = 1080  # lidar size
NO_VAE_VAR = True

BLOCK_SIZE = 32  # sequence length (context)

class EnvEncoder(object):
    """ Generic class to encode the observations of an environment,
    look at EncodedEnv to see how it is typically used """
    def __init__(self,
                 backend, encoding,
                 rnn_model_path=os.path.expanduser("~/navrep/models/M/rnn.json"),
                 rnn1d_model_path=os.path.expanduser("~/navrep/models/M/rnn1d.json"),
                 vae_model_path=os.path.expanduser("~/navrep/models/V/vae.json"),
                 vae1d_model_path=os.path.expanduser("~/navrep/models/V/vae1d.json"),
                 gpt_model_path=os.path.expanduser("~/navrep/models/W/gpt"),
                 gpt1d_model_path=os.path.expanduser("~/navrep/models/W/gpt1d"),
                 vae1dlstm_model_path=os.path.expanduser("~/navrep/models/W/vae1dlstm"),
                 vaelstm_model_path=os.path.expanduser("~/navrep/models/W/vaelstm"),
                 gpu=False,
                 encoder_to_share_model_with=None,  # another EnvEncoder
                 ):
        LIDAR_NORM_FACTOR = None
        if backend == "GPT":
            from navrep.scripts.train_gpt import _Z, _H
        elif backend == "GPT1D":
            from navrep.scripts.train_gpt1d import _Z, _H
            from navrep.tools.wdataset import LIDAR_NORM_FACTOR
        elif backend == "VAE1DLSTM":
            from navrep.scripts.train_vae1dlstm import _Z, _H
            from navrep.tools.wdataset import LIDAR_NORM_FACTOR
        elif backend == "VAELSTM":
            from navrep.scripts.train_vaelstm import _Z, _H
        elif backend == "VAE_LSTM":
            from navrep.scripts.train_vae import _Z
            from navrep.scripts.train_rnn import _H
        elif backend == "VAE1D_LSTM":
            from navrep.scripts.train_vae1d import _Z
            from navrep.scripts.train_rnn import _H
            from navrep.scripts.train_vae1d import MAX_LIDAR_DIST as LIDAR_NORM_FACTOR
        self._Z = _Z
        self._H = _H
        self.LIDAR_NORM_FACTOR = LIDAR_NORM_FACTOR
        self.encoding = encoding
        self.backend = backend
        if self.encoding == "V_ONLY":
            self.encoding_dim = _Z + _RS
        elif self.encoding == "VM":
            self.encoding_dim = _Z + _H + _RS
        elif self.encoding == "M_ONLY":
            self.encoding_dim = _H + _RS
        else:
            raise NotImplementedError
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self.encoding_dim,), dtype=np.float32)
        # V + M Models
        if encoder_to_share_model_with is not None:
            self.vae = encoder_to_share_model_with.vae
            self.rnn = encoder_to_share_model_with.rnn
        else:
            # load world model
            if self.backend == "VAE_LSTM":
                reset_graph()
                self.vae = ConvVAE(z_size=_Z, batch_size=1, is_training=False)
                self.vae.load_json(vae_model_path)
                if self.encoding in ["VM", "M_ONLY"]:
                    hps = sample_hps_params. _replace(seq_width=_Z+_G, action_width=_A, rnn_size=_H)
                    self.rnn = MDNRNN(hps, gpu_mode=gpu)
                    self.rnn.load_json(rnn_model_path)
            elif self.backend == "VAE1D_LSTM":
                reset_graph()
                self.vae = Conv1DVAE(z_size=_Z, batch_size=1, is_training=False)
                self.vae.load_json(vae1d_model_path)
                if self.encoding in ["VM", "M_ONLY"]:
                    hps = sample_hps_params. _replace(seq_width=_Z+_G, action_width=_A, rnn_size=_H)
                    self.rnn = MDNRNN(hps, gpu_mode=gpu)
                    self.rnn.load_json(rnn1d_model_path)
            elif self.backend == "GPT":
                mconf = GPTConfig(BLOCK_SIZE, _H)
                model = GPT(mconf, gpu=gpu)
                load_checkpoint(model, gpt_model_path, gpu=gpu)
                self.vae = model
                self.rnn = model
            elif self.backend == "GPT1D":
                mconf = GPTConfig(BLOCK_SIZE, _H)
                model = GPT1D(mconf, gpu=gpu)
                load_checkpoint(model, gpt1d_model_path, gpu=gpu)
                self.vae = model
                self.rnn = model
            elif self.backend == "VAELSTM":
                mconf = VAELSTMConfig(_Z, _H)
                model = VAELSTM(mconf, gpu=gpu)
                load_checkpoint(model, vaelstm_model_path, gpu=gpu)
                self.vae = model
                self.rnn = model
            elif self.backend == "VAE1DLSTM":
                mconf = VAE1DLSTMConfig(_Z, _H)
                model = VAE1DLSTM(mconf, gpu=gpu)
                load_checkpoint(model, vae1dlstm_model_path, gpu=gpu)
                self.vae = model
                self.rnn = model
            else:
                raise NotImplementedError
        # other tools
        self.rings_def = generate_rings(_64, _64)
        self.viewer = None
        # environment state variables
        self.reset()

    def reset(self):
        if self.encoding in ["VM", "M_ONLY"]:
            if self.backend in ["VAE_LSTM", "VAE1D_LSTM"]:
                self.state = rnn_init_state(self.rnn)
            elif self.backend in ["GPT", "VAELSTM", "VAE1DLSTM", "GPT1D"]:
                self.gpt_sequence = []
        self.lidar_z = np.zeros(self._Z)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()

    def _get_last_decoded_scan(self):
        obs_pred = self.vae.decode(self.lidar_z.reshape((1,self._Z)))
        if self.backend in ["VAE1DLSTM", "GPT1D", "VAE1D_LSTM"]:
            decoded_scan = (obs_pred * self.LIDAR_NORM_FACTOR).reshape((_L))
        else:
            rings_pred = obs_pred * self.rings_def["rings_to_bool"]
            decoded_scan = self.rings_def["rings_to_lidar"](rings_pred, _L).reshape((_L))
        return decoded_scan

    def _encode_obs(self, obs, action):
        """
    obs is (lidar, other_obs)
    where lidar is (time_samples, ray, channel)
    and other_obs is (5,) - [goal_x, goal_y, vel_x, vel_y, vel_theta] all in robot frame

    h is (32+2+512), i.e. concat[lidar_z, robotstate, h rnn state]
    lidar_z is -inf, inf
    h rnn state is ?
    other_obs is -inf, inf
    """
        # convert lidar scan to obs
        lidar_scan = obs[0]  # latest scan only obs (buffer, ray, channel)
        lidar_scan = lidar_scan.reshape(1, _L).astype(np.float32)
        lidar_mode = "scans" if "1D" in self.backend else "rings"
        lidar_obs = scans_to_lidar_obs(lidar_scan, lidar_mode, self.rings_def, channel_first=False)
        self.last_lidar_obs = lidar_obs  # for rendering purposes

        # obs to z, mu, logvar
        mu, logvar = self.vae.encode_mu_logvar(lidar_obs)
        mu = mu[0]
        logvar = logvar[0]
        s = logvar.shape
        if NO_VAE_VAR:
            lidar_z = mu * 1.
        else:
            lidar_z = mu + np.exp(logvar / 2.0) * np.random.randn(*s)

        # encode obs through V + M
        self.lidar_z = lidar_z
        if self.encoding == "V_ONLY":
            encoded_obs = np.concatenate([self.lidar_z, obs[1]], axis=0)
        elif self.encoding in ["VM", "M_ONLY"]:
            # get h
            if self.backend in ["VAE_LSTM", "VAE1D_LSTM"]:
                goal_z = obs[1][:2] / MAX_GOAL_DIST
                rnn_z = np.concatenate([lidar_z, goal_z], axis=-1)
                self.state = rnn_next_state(self.rnn, rnn_z, action, self.state)
                h = self.state.h[0]
            elif self.backend in ["GPT", "VAELSTM", "VAE1DLSTM", "GPT1D"]:
                self.gpt_sequence.append(dict(obs=lidar_obs[0], state=obs[1][:2], action=action))
                self.gpt_sequence = self.gpt_sequence[:BLOCK_SIZE]
                h = self.rnn.get_h(self.gpt_sequence)
            # encoded obs
            if self.encoding == "VM":
                encoded_obs = np.concatenate([self.lidar_z, obs[1], h], axis=0)
            elif self.encoding == "M_ONLY":
                encoded_obs = np.concatenate([obs[1], h], axis=0)
        return encoded_obs

    def _render_rings_polar(self, close, save_to_file=False):
        if close:
            self.viewer.close()
            return
        # rendering
        if self.backend in ["VAE1DLSTM", "GPT1D", "VAE1D_LSTM"]:
            return False
        else:
            last_rings_obs = self.last_lidar_obs.reshape((_64, _64, 1))
            last_rings_pred = self.vae.decode(self.lidar_z.reshape((1,self._Z))).reshape((_64, _64, 1))
            import matplotlib.pyplot as plt
            plt.ion()
            fig, (ax1, ax2) = plt.subplots(
                1, 2, subplot_kw=dict(projection="polar"), num="rings"
            )
            ax1.clear()
            ax2.clear()
            if self.viewer is None:
                self.rendering_iteration = 0
            self.viewer = fig
            self.rings_def["visualize_rings"](last_rings_obs, scan=None, fig=fig, ax=ax1)
            self.rings_def["visualize_rings"](last_rings_pred, scan=None, fig=fig, ax=ax2)
            ax1.set_ylim([0, 10])
            ax1.set_title("ground truth")
            ax2.set_ylim([0, 10])
            ax2.set_title("lidar reconstruction")
            # rings box viz
            fig2, (ax1, ax2) = plt.subplots(1, 2, num="2d")
            ax1.clear()
            ax2.clear()
            ax1.imshow(np.squeeze(last_rings_obs), cmap=plt.cm.Greys)
            ax2.imshow(np.squeeze(last_rings_pred), cmap=plt.cm.Greys)
            ax1.set_title("ground truth")
            ax2.set_title("lidar reconstruction")
            # update
            plt.pause(0.01)
            self.rendering_iteration += 1
            if save_to_file:
                fig.savefig(
                    "/tmp/encodedenv_polar{:04d}.png".format(self.rendering_iteration))
                fig2.savefig(
                    "/tmp/encodedenv_box{:04d}.png".format(self.rendering_iteration))

    def _render_rings(self, close, save_to_file=False):
        if close:
            self.viewer.close()
            return
        # rendering
        if self.backend in ["VAE1DLSTM", "GPT1D", "VAE1D_LSTM"]:
            return False
        else:
            last_rings_obs = self.last_lidar_obs.reshape((_64, _64))
            last_rings_pred = self.vae.decode(self.lidar_z.reshape((1,self._Z))).reshape((_64, _64))
            # Window and viewport size
            ring_size = _64  # grid cells
            padding = 4  # grid cells
            grid_size = 1  # px per grid cell
            WINDOW_W = (2 * ring_size + 3 * padding) * grid_size
            WINDOW_H = (1 * ring_size + 2 * padding) * grid_size
            VP_W = WINDOW_W
            VP_H = WINDOW_H
            from gym.envs.classic_control import rendering
            import pyglet
            from pyglet import gl
            # Create viewer
            if self.viewer is None:
                self.viewer = rendering.Viewer(WINDOW_W, WINDOW_H)
                self.rendering_iteration = 0
            # Render in pyglet
            win = self.viewer.window
            win.switch_to()
            win.dispatch_events()
            win.clear()
            gl.glViewport(0, 0, VP_W, VP_H)
            # colors
            bgcolor = np.array([0.4, 0.8, 0.4])
            # Green background
            gl.glBegin(gl.GL_QUADS)
            gl.glColor4f(bgcolor[0], bgcolor[1], bgcolor[2], 1.0)
            gl.glVertex3f(0, VP_H, 0)
            gl.glVertex3f(VP_W, VP_H, 0)
            gl.glVertex3f(VP_W, 0, 0)
            gl.glVertex3f(0, 0, 0)
            gl.glEnd()
            # rings - observation
            w_offset = 0
            for rings in [last_rings_obs, last_rings_pred]:
                for i in range(ring_size):
                    for j in range(ring_size):
                        cell_color = 1 - rings[i, j]
                        cell_y = (padding + i) * grid_size  # px
                        cell_x = (padding + j + w_offset) * grid_size  # px
                        gl.glBegin(gl.GL_QUADS)
                        gl.glColor4f(cell_color, cell_color, cell_color, 1.0)
                        gl.glVertex3f(cell_x+       0,  cell_y+grid_size, 0)  # noqa
                        gl.glVertex3f(cell_x+grid_size, cell_y+grid_size, 0)  # noqa
                        gl.glVertex3f(cell_x+grid_size, cell_y+        0, 0)  # noqa
                        gl.glVertex3f(cell_x+        0, cell_y+        0, 0)  # noqa
                        gl.glEnd()
                w_offset += ring_size + padding
            if save_to_file:
                pyglet.image.get_buffer_manager().get_color_buffer().save(
                    "/tmp/encodeder_rings{:04d}.png".format(self.rendering_iteration))
            # actualize
            win.flip()
            self.rendering_iteration += 1
            return self.viewer.isopen

class EncodedEnv(IANEnv):
    """ takes a (2) action as input
    outputs encoded obs (546) """
    def __init__(self, backend, encoding,
                 silent=False, max_episode_length=1000, collect_trajectories=False,
                 gpu=False, encoder=None):
        if encoder is None:
            encoder = EnvEncoder(backend, encoding)
        self.encoder = encoder
        super(EncodedEnv, self).__init__(
            silent=silent, max_episode_length=max_episode_length, collect_trajectories=collect_trajectories)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = self.encoder.observation_space

    def step(self, action):
        action = np.array([action[0], action[1], 0.])  # no rotation
        obs, reward, done, info = super(EncodedEnv, self).step(action)
        h = self.encoder._encode_obs(obs, action)
        return h, reward, done, info

    def reset(self):
        self.encoder.reset()
        obs = super(EncodedEnv, self).reset()
        h = self.encoder._encode_obs(obs, np.array([0,0,0]))
        return h

    def render(self, mode="human", close=False, save_to_file=False, robocentric=False,
               render_decoded_scan=True):
        if mode == "rings":
            self.encoder._render_rings(close=close)
            return
        decoded_scan = None
        if render_decoded_scan:
            decoded_scan = self.encoder._get_last_decoded_scan()
        super(EncodedEnv, self).render(mode=mode, close=close, lidar_scan_override=decoded_scan,
                                       save_to_file=save_to_file, robocentric=robocentric)
