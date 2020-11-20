from __future__ import print_function
import numpy as np
import os
import threading

from navrep.models.rnn import reset_graph, sample_hps_params, MDNRNN, get_pi_idx
from navrep.models.vae2d import ConvVAE

_64 = 64  # size of VAE image input
_Z = 32  # size of VAE z latent space

class ImDreamEnv(object):
    def __init__(self, temperature=0.25):
        # constants
        self.TEMPERATURE = temperature
        self.DT = 0.2  # should be the same as data rnn was trained with
        initial_z_path = os.path.expanduser(
            "~/navrep/datasets/M/im/corridor_koze_kids_bag_mus_logvars_robotstates_actions_rewards_dones.npz"
        )
        rnn_model_path = os.path.expanduser("~/navrep/models/M/imrnn.json")
        vae_model_path = os.path.expanduser("~/navrep/models/V/imvae.json")
        # V + M Models
        reset_graph()
        self.rnn = MDNRNN(sample_hps_params, gpu_mode=False)
        self.vae = ConvVAE(batch_size=1, is_training=False, channels=3)
        self.vae.load_json(vae_model_path)
        self.rnn.load_json(rnn_model_path)
        # load initial image encoding
        arrays = np.load(initial_z_path)
        initial_mu = arrays["mus"][0]
        initial_logvar = arrays["logvars"][0]
        self.initial_z = initial_mu + np.exp(initial_logvar / 2.0) * np.random.randn(
            *(initial_mu.shape)
        )
        # other tools
        self.viewer = None
        # environment state variables
        self.reset()
        # hot-start the rnn state
        for i in range(20):
            self.step(np.array([0,0,0]), override_next_z=self.initial_z)

    def step(self, action, override_next_z=None):
        feed = {
            self.rnn.input_z: np.reshape(self.prev_z, (1, 1, _Z)),
            self.rnn.input_action: np.reshape(action, (1, 1, 3)),
            self.rnn.input_restart: np.reshape(self.prev_restart, (1, 1)),
            self.rnn.initial_state: self.rnn_state,
        }

        [logmix, mean, logstd, logrestart, next_state] = self.rnn.sess.run(
            [
                self.rnn.out_logmix,
                self.rnn.out_mean,
                self.rnn.out_logstd,
                self.rnn.out_restart_logits,
                self.rnn.final_state,
            ],
            feed,
        )
        OUTWIDTH = _Z

        if self.TEMPERATURE == 0:  # deterministically pick max of MDN distribution
            mixture_idx = np.argmax(logmix, axis=-1)
            chosen_mean = mean[(range(OUTWIDTH), mixture_idx)]
            chosen_logstd = logstd[(range(OUTWIDTH), mixture_idx)]
            next_z = chosen_mean
        else:  # sample from modelled MDN distribution
            mixprob = np.copy(logmix) / self.TEMPERATURE  # adjust temperatures
            mixprob -= mixprob.max()
            mixprob = np.exp(mixprob)
            mixprob /= mixprob.sum(axis=1).reshape(OUTWIDTH, 1)

            mixture_idx = np.zeros(OUTWIDTH)
            chosen_mean = np.zeros(OUTWIDTH)
            chosen_logstd = np.zeros(OUTWIDTH)
            for j in range(OUTWIDTH):
                idx = get_pi_idx(np.random.rand(), mixprob[j])
                mixture_idx[j] = idx
                chosen_mean[j] = mean[j][idx]
                chosen_logstd[j] = logstd[j][idx]
            rand_gaussian = np.random.randn(OUTWIDTH) * np.sqrt(self.TEMPERATURE)
            next_z = chosen_mean + np.exp(chosen_logstd) * rand_gaussian
        if sample_hps_params.differential_z:
            next_z = self.prev_z + next_z

        next_restart = 0
        #         if logrestart[0] > 0:
        #             next_restart = 1

        self.prev_z = next_z
        if override_next_z is not None:
            self.prev_z = override_next_z
        self.prev_restart = next_restart
        self.rnn_state = next_state
        # logging-only vars, used for rendering
        self.prev_action = action
        self.episode_step += 1

        return next_z, None, next_restart, {}

    def reset(self):
        self.prev_z = self.initial_z
        self.prev_restart = np.array([1])
        self.rnn_state = self.rnn.sess.run(self.rnn.zero_state)
        # logging vars
        self.prev_action = np.array([0.0, 0.0, 0.0])
        self.episode_step = 0

    def render(self, mode="human", close=False):
        img_pred = (self.vae.decode(self.prev_z.reshape(1, _Z)) * 255).astype(np.uint8)
        img_pred = img_pred.reshape(_64, _64, 3)

        if mode == "rgb_array":
            raise NotImplementedError
        elif mode == "human":
            # Window and viewport size
            WINDOW_W = 256
            WINDOW_H = 256
            VP_W = WINDOW_W
            VP_H = WINDOW_H
            from gym.envs.classic_control import rendering
            import pyglet
            from pyglet import gl

            # Create pyglet image
#             pixels = [
#                 255, 0, 0,      0, 255, 0,      0, 0, 255,     # RGB values range from
#                 255, 0, 0,      255, 0, 0,      255, 0, 0,     # 0 to 255 for each color
#                 255, 0, 0,      255, 0, 0,      255, 0, 0,     # component.
#             ]
            from pyglet.gl.gl import GLubyte
            pixels = img_pred.flatten()
            rawData = (GLubyte * len(pixels))(*pixels)
            image_data = pyglet.image.ImageData(_64, _64, 'RGB', rawData)

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
            with self.image_lock:
                self.currently_rendering_iteration += 1
                self.viewer.draw_circle(r=10, color=(0.3, 0.3, 0.3))
                win = self.viewer.window
                win.switch_to()
                win.dispatch_events()
                win.clear()
                gl.glViewport(0, 0, VP_W, VP_H)
                # Image
                image_data.blit(96,96)
                # Text
                self.score_label.text = "A {:.1f} {:.1f} {:.1f} S {}".format(
                    self.prev_action[0],
                    self.prev_action[1],
                    self.prev_action[2],
                    self.episode_step,
                )
                self.score_label.draw()
                win.flip()
                return self.viewer.isopen

