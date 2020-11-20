from __future__ import print_function
import numpy as np
import os
import threading

from navrep.tools.rings import generate_rings
from navrep.models.tcn import reset_graph, sample_hps_params, MDNTCN, get_pi_idx
from navrep.models.vae2d import ConvVAE

_Z = 32

class ToyTCNDreamEnv(object):
    def __init__(self, temperature=0.25):
        # constants
        self.TEMPERATURE = temperature
        self.SEQLEN = 99
        self.DT = 0.2  # should be the same as data rnn was trained with
        initial_z_path = os.path.expanduser(
            "~/navrep/datasets/M/toy/000_mus_logvars_robotstates_actions_rewards_dones.npz"
        )
        tcn_model_path = os.path.expanduser("~/navrep/models/M/toytcn.json")
        vae_model_path = os.path.expanduser("~/navrep/models/V/toyvae.json")
        # V + M Models
        reset_graph()
        params = sample_hps_params._replace(max_seq_len=self.SEQLEN+1)
        self.tcn = MDNTCN(params, gpu_mode=False)
        self.vae = ConvVAE(batch_size=1, is_training=False)
        self.vae.load_json(vae_model_path)
        self.tcn.load_json(tcn_model_path)
        # load initial image encoding
        arrays = np.load(initial_z_path)
        # other tools
        self.rings_def = generate_rings(64, 64)
        self.viewer = None
        # environment state variables
        self.reset()
        # hot-start the tcn state
        self.sequence_z = arrays["mus"][:self.SEQLEN].reshape((1, self.SEQLEN, _Z))
        self.sequence_action = arrays["actions"][:self.SEQLEN].reshape((1, self.SEQLEN, 3))
        self.sequence_restart = arrays["dones"][:self.SEQLEN].reshape((1, self.SEQLEN))

    def step(self, action, override_next_z=None):
        # predict for fixed-sized sequence, lpadded with zeros
        self.sequence_action[0, -1, :] = action
        feed = {
            self.tcn.input_z: np.reshape(self.sequence_z[:self.SEQLEN], (1, self.SEQLEN, _Z)),
            self.tcn.input_action: np.reshape(self.sequence_action[:self.SEQLEN], (1, self.SEQLEN, 3)),
            self.tcn.input_restart: np.reshape(self.sequence_restart[:self.SEQLEN], (1, self.SEQLEN)),
        }

        [logmix, mean, logstd, logrestart] = self.tcn.sess.run(
            [self.tcn.out_logmix, self.tcn.out_mean, self.tcn.out_logstd, self.tcn.out_restart_logits], feed
        )

        logmix = logmix.reshape((self.SEQLEN, _Z, sample_hps_params.num_mixture))
        mean = mean.reshape((self.SEQLEN, _Z, sample_hps_params.num_mixture))
        logstd = logstd.reshape((self.SEQLEN, _Z, sample_hps_params.num_mixture))
        logrestart = logrestart.reshape((self.SEQLEN, 1))

        OUTWIDTH = _Z

        # adjust temperatures
        logmix2 = np.copy(logmix) / self.TEMPERATURE
        logmix2 -= logmix2.max()
        logmix2 = np.exp(logmix2)
        logmix2 /= logmix2.sum(axis=-1).reshape((self.SEQLEN, _Z, 1))

        mixture_idx = np.zeros((self.SEQLEN, OUTWIDTH))
        chosen_mean = np.zeros((self.SEQLEN, OUTWIDTH))
        chosen_logstd = np.zeros((self.SEQLEN, OUTWIDTH))
        for i in range(len(mixture_idx)):
            for j in range(OUTWIDTH):
                idx = get_pi_idx(np.random.rand(), logmix2[i, j])
                mixture_idx[i, j] = idx
                chosen_mean[i, j] = mean[i, j][idx]
                chosen_logstd[i, j] = logstd[i, j][idx]

        rand_gaussian = np.random.randn(self.SEQLEN, OUTWIDTH) * np.sqrt(self.TEMPERATURE)
        seq_z_predicted = chosen_mean + np.exp(chosen_logstd) * rand_gaussian
        if sample_hps_params.differential_z:
            seq_z_predicted = np.reshape(self.sequence_z[:self.SEQLEN], (1, self.SEQLEN, _Z)) + seq_z_predicted

        # pick last output
        next_z = seq_z_predicted[0, -1, :]

        next_restart = 0
        #         if logrestart[0] > 0:
        #             next_restart = 1

        # update variables
        self.sequence_z[0, :-1, :] = self.sequence_z[0, 1:, :]
        self.sequence_action[0, :-1, :] = self.sequence_action[0, 1:, :]
        self.sequence_restart[0, :-1] = self.sequence_restart[0, 1:]
        self.sequence_z[0, -1, :] = next_z
        self.sequence_action[0, -1, :] = np.nan
        self.sequence_restart[0, -1] = next_restart

        # logging-only vars, used for rendering
        self.prev_action = action
        self.episode_step += 1

        return next_z, None, next_restart, {}

    def reset(self):
        # logging vars
        self.prev_action = np.array([0.0, 0.0, 0.0])
        self.episode_step = 0

    def render(self, mode="human", close=False):
        rings_pred = (
            self.vae.decode(self.sequence_z[0, -1].reshape(1, _Z))
            * self.rings_def["rings_to_bool"]
        )
        predicted_ranges = self.rings_def["rings_to_lidar"](rings_pred, 1080)

        if mode == "rgb_array":
            raise NotImplementedError
        elif mode == "human":
            # Window and viewport size
            WINDOW_W = 256
            WINDOW_H = 256
            M_PER_PX = 25.6 / WINDOW_H
            VP_W = WINDOW_W
            VP_H = WINDOW_H
            from gym.envs.classic_control import rendering
            import pyglet
            from pyglet import gl

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
            def make_circle(c, r, res=10):
                thetas = np.linspace(0, 2 * np.pi, res + 1)[:-1]
                verts = np.zeros((res, 2))
                verts[:, 0] = c[0] + r * np.cos(thetas)
                verts[:, 1] = c[1] + r * np.sin(thetas)
                return verts

            with self.image_lock:
                self.currently_rendering_iteration += 1
                self.viewer.draw_circle(r=10, color=(0.3, 0.3, 0.3))
                win = self.viewer.window
                win.switch_to()
                win.dispatch_events()
                win.clear()
                gl.glViewport(0, 0, VP_W, VP_H)
                # colors
                bgcolor = np.array([0.4, 0.8, 0.4])
                nosecolor = np.array([0.3, 0.3, 0.3])
                lidarcolor = np.array([1.0, 0.0, 0.0])
                # Green background
                gl.glBegin(gl.GL_QUADS)
                gl.glColor4f(bgcolor[0], bgcolor[1], bgcolor[2], 1.0)
                gl.glVertex3f(0, VP_H, 0)
                gl.glVertex3f(VP_W, VP_H, 0)
                gl.glVertex3f(VP_W, 0, 0)
                gl.glVertex3f(0, 0, 0)
                gl.glEnd()
                # LIDAR
                i = WINDOW_W / 2.0
                j = WINDOW_H / 2.0
                angle = np.pi / 2.0
                scan = np.squeeze(predicted_ranges)
                lidar_angles = np.linspace(0, 2 * np.pi, len(scan) + 1)[:-1]
                i_ray_ends = i + scan / M_PER_PX * np.cos(lidar_angles)
                j_ray_ends = j + scan / M_PER_PX * np.sin(lidar_angles)
                is_in_fov = np.cos(lidar_angles - angle) >= 0.78
                for ray_idx in range(len(scan)):
                    end_i = i_ray_ends[ray_idx]
                    end_j = j_ray_ends[ray_idx]
                    gl.glBegin(gl.GL_LINE_LOOP)
                    if is_in_fov[ray_idx]:
                        gl.glColor4f(1.0, 1.0, 0.0, 0.1)
                    else:
                        gl.glColor4f(lidarcolor[0], lidarcolor[1], lidarcolor[2], 0.1)
                    gl.glVertex3f(i, j, 0)
                    gl.glVertex3f(end_i, end_j, 0)
                    gl.glEnd()
                # Agent body
                i = WINDOW_W / 2.0
                j = WINDOW_H / 2.0
                r = 0.3 / M_PER_PX
                angle = np.pi / 2.0
                poly = make_circle((i, j), r)
                gl.glBegin(gl.GL_POLYGON)
                color = np.array([1.0, 1.0, 1.0])
                gl.glColor4f(color[0], color[1], color[2], 1)
                for vert in poly:
                    gl.glVertex3f(vert[0], vert[1], 0)
                gl.glEnd()
                # Direction triangle
                inose = i + r * np.cos(angle)
                jnose = j + r * np.sin(angle)
                iright = i + 0.3 * r * -np.sin(angle)
                jright = j + 0.3 * r * np.cos(angle)
                ileft = i - 0.3 * r * -np.sin(angle)
                jleft = j - 0.3 * r * np.cos(angle)
                gl.glBegin(gl.GL_TRIANGLES)
                gl.glColor4f(nosecolor[0], nosecolor[1], nosecolor[2], 1)
                gl.glVertex3f(inose, jnose, 0)
                gl.glVertex3f(iright, jright, 0)
                gl.glVertex3f(ileft, jleft, 0)
                gl.glEnd()
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
