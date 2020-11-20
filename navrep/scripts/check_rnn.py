from __future__ import print_function
import numpy as np
import os
import json
import tensorflow as tf
import random
import time
from timeit import default_timer as timer

from navrep.tools.rings import generate_rings
from navrep.models.rnn import reset_graph, sample_hps_params, MDNRNN, get_pi_idx
from navrep.models.vae2d import ConvVAE

# parameters
TEMPERATURE = 0.5
_Z = 32

sequence_z_path = os.path.expanduser(
    "~/navrep/datasets/M/ian/000_mus_logvars_robotstates_actions_rewards_dones.npz"
)
rnn_model_path = os.path.expanduser("~/navrep/models/M/rnn.json")
vae_model_path = os.path.expanduser("~/navrep/models/V/vae.json")

reset_graph()
rnn = MDNRNN(sample_hps_params, gpu_mode=False)
vae = ConvVAE(batch_size=1, is_training=False)

vae.load_json(vae_model_path)
rnn.load_json(rnn_model_path)

rings_def = generate_rings(64, 64)

# load sequence image encoding
arrays = np.load(sequence_z_path)
sequence_action = arrays["actions"]
sequence_mu = arrays["mus"]
sequence_logvar = arrays["logvars"]
sequence_z = sequence_mu + np.exp(sequence_logvar / 2.0) * np.random.randn(
    *(sequence_mu.shape)
)
SEQUENCE_LENGTH = len(sequence_mu)

prev_z = sequence_z[0]
prev_z_predicted = sequence_z[0]
prev_action = sequence_action[0]
prev_restart = np.array([0])
rnn_state = rnn.sess.run(rnn.zero_state)
for i in range(SEQUENCE_LENGTH):
    print(prev_action)

    feed = {
        rnn.input_z: np.reshape(prev_z, (1, 1, _Z)),
        rnn.input_action: np.reshape(prev_action, (1, 1, 3)),
        rnn.input_restart: np.reshape(prev_restart, (1, 1)),
        rnn.initial_state: rnn_state,
    }

    [logmix, mean, logstd, logrestart, next_state] = rnn.sess.run(
        [
            rnn.out_logmix,
            rnn.out_mean,
            rnn.out_logstd,
            rnn.out_restart_logits,
            rnn.final_state,
        ],
        feed,
    )

    OUTWIDTH = _Z

    if TEMPERATURE == 0:  # deterministically pick max of MDN distribution
        mixture_idx = np.argmax(logmix, axis=-1)
        chosen_mean = mean[(range(OUTWIDTH), mixture_idx)]
        chosen_logstd = logstd[(range(OUTWIDTH), mixture_idx)]
        next_z_predicted = chosen_mean
    else:  # sample from modelled MDN distribution
        logmix2 = np.copy(logmix) / TEMPERATURE  # adjust temperatures
        logmix2 -= logmix2.max()
        logmix2 = np.exp(logmix2)
        logmix2 /= logmix2.sum(axis=1).reshape(OUTWIDTH, 1)

        mixture_idx = np.zeros(OUTWIDTH)
        chosen_mean = np.zeros(OUTWIDTH)
        chosen_logstd = np.zeros(OUTWIDTH)
        for j in range(OUTWIDTH):
            idx = get_pi_idx(np.random.rand(), logmix2[j])
            mixture_idx[j] = idx
            chosen_mean[j] = mean[j][idx]
            chosen_logstd[j] = logstd[j][idx]

        rand_gaussian = np.random.randn(OUTWIDTH) * np.sqrt(TEMPERATURE)
        next_z_predicted = chosen_mean + np.exp(chosen_logstd) * rand_gaussian
    if sample_hps_params.differential_z:
        next_z_predicted = prev_z + next_z_predicted  # if in the residual regime

    # take z from sequence
    next_z = sequence_z[i]
    prev_action = sequence_action[i]

    next_restart = 0
    done = False
    if logrestart[0] > 0:
        next_restart = 1
        done = True

    rings_pred = vae.decode(prev_z.reshape(1, _Z)) * rings_def["rings_to_bool"]
    rings_z_pred_pred = (
        vae.decode(prev_z_predicted.reshape(1, _Z)) * rings_def["rings_to_bool"]
    )
    predicted_ranges = rings_def["rings_to_lidar"](rings_pred, 1080)
    if True:
        import matplotlib.pyplot as plt

        plt.ion()
        plt.figure("rings")
        plt.clf()
        fig, (ax1, ax2) = plt.subplots(
            2, 1, subplot_kw=dict(projection="polar"), num="rings"
        )
        rings_def["visualize_rings"](rings_pred[0], scan=None, fig=fig, ax=ax1)
        rings_def["visualize_rings"](rings_z_pred_pred[0], scan=None, fig=fig, ax=ax2)
        ax1.set_ylim([0, 10])
        ax1.set_title(i)
        ax2.set_ylim([0, 10])
        # update
        plt.pause(0.01)

    prev_z = next_z
    prev_z_predicted = next_z_predicted
    prev_restart = next_restart
    rnn_state = next_state
