from __future__ import print_function
import numpy as np
import os
import threading

from navrep.tools.rings import generate_rings
from navrep.models.rnn import reset_graph, sample_hps_params, MDNRNN, get_pi_idx, MAX_GOAL_DIST
from navrep.models.vae2d import ConvVAE

_Z = 32
_G = 2

class DreamEnv(object):
    def __init__(self, temperature=0.25,
                 initial_z_path=os.path.expanduser(
                     "~/navrep/datasets/M/ian/000_mus_logvars_robotstates_actions_rewards_dones.npz"
                 ),
                 rnn_model_path=os.path.expanduser("~/navrep/models/M/rnn.json"),
                 vae_model_path=os.path.expanduser("~/navrep/models/V/vae.json"),
                 ):
        # constants
        self.TEMPERATURE = temperature
        self.DT = 0.5  # should be the same as data rnn was trained with
        # V + M Models
        reset_graph()
        self.rnn = MDNRNN(sample_hps_params, gpu_mode=False)
        self.vae = ConvVAE(batch_size=1, is_training=False)
        self.vae.load_json(vae_model_path)
        self.rnn.load_json(rnn_model_path)
        # load initial image encoding
        arrays = np.load(initial_z_path)
        initial_mu = arrays["mus"][0]
        initial_logvar = arrays["logvars"][0]
        initial_robotstate = arrays["robotstates"][0]
        ini_lidar_z = initial_mu + np.exp(initial_logvar / 2.0) * np.random.randn(
            *(initial_mu.shape)
        )
        ini_goal_z = initial_robotstate[:2] / MAX_GOAL_DIST
        self.initial_z = np.concatenate([ini_lidar_z, ini_goal_z], axis=-1)
        # other tools
        self.rings_def = generate_rings(64, 64)
        self.viewer = None
        # environment state variables
        self.reset()
        # hot-start the rnn state
        for i in range(20):
            self.step(np.array([0,0,0]), override_next_z=self.initial_z)

    def step(self, action, override_next_z=None):
        feed = {
            self.rnn.input_z: np.reshape(self.prev_z, (1, 1, _Z+_G)),
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
        OUTWIDTH = _Z+_G

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
        if close:
            if self.viewer is not None:
                self.viewer.close()
            return

        # get last z decoding
        rings_pred = (
            self.vae.decode(self.prev_z.reshape(1, _Z+_G)[:, :_Z])
            * self.rings_def["rings_to_bool"]
        )
        predicted_ranges = self.rings_def["rings_to_lidar"](rings_pred, 1080)
        goal_pred = self.prev_z.reshape((_Z+_G,))[_Z:] * MAX_GOAL_DIST

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
                lidar_angles = lidar_angles + np.pi / 2.  # make robot face up
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
                # Goal
                goalcolor = np.array([1., 1., 0.3])
                px_goal = goal_pred / M_PER_PX
                igoal = i - px_goal[1]  # rotate 90deg to face up
                jgoal = j + px_goal[0]
                # Goal line
                gl.glBegin(gl.GL_LINE_LOOP)
                gl.glColor4f(goalcolor[0], goalcolor[1], goalcolor[2], 1)
                gl.glVertex3f(i, j, 0)
                gl.glVertex3f(igoal, jgoal, 0)
                gl.glEnd()
                # Goal markers
                gl.glBegin(gl.GL_TRIANGLES)
                gl.glColor4f(goalcolor[0], goalcolor[1], goalcolor[2], 1)
                triangle = make_circle((igoal, jgoal), r/3., res=3)
                for vert in triangle:
                    gl.glVertex3f(vert[0], vert[1], 0)
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

    def close(self):
        self.render(close=True)

    def _get_dt(self):
        return self.DT

    def _get_viewer(self):
        return self.viewer
