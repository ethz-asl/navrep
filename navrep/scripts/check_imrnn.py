from __future__ import print_function
import numpy as np
import os

from navrep.models.rnn import reset_graph, sample_hps_params, MDNRNN, get_pi_idx
from navrep.models.vae2d import ConvVAE

# parameters
TEMPERATURE = 0.5
_Z = 32

sequence_z_path = os.path.expanduser(
    "~/navrep/datasets/M/im/corridor_koze_kids_bag_mus_logvars_actions_rewards_dones.npz"
)
rnn_model_path = os.path.expanduser("~/navrep/models/M/imrnn.json")
vae_model_path = os.path.expanduser("~/navrep/models/V/imvae.json")

reset_graph()
imrnn = MDNRNN(sample_hps_params, gpu_mode=False)
imvae = ConvVAE(batch_size=1, is_training=False, channels=3)

imvae.load_json(vae_model_path)
imrnn.load_json(rnn_model_path)

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
rnn_state = imrnn.sess.run(imrnn.zero_state)
for i in range(SEQUENCE_LENGTH):
    print(prev_action)

    feed = {
        imrnn.input_z: np.reshape(prev_z, (1, 1, _Z)),
        imrnn.input_action: np.reshape(prev_action, (1, 1, 3)),
        imrnn.input_restart: np.reshape(prev_restart, (1, 1)),
        imrnn.initial_state: rnn_state,
    }

    [logmix, mean, logstd, logrestart, next_state] = imrnn.sess.run(
        [
            imrnn.out_logmix,
            imrnn.out_mean,
            imrnn.out_logstd,
            imrnn.out_restart_logits,
            imrnn.final_state,
        ],
        feed,
    )

    OUTWIDTH = _Z

    # adjust temperatures
    logmix2 = np.copy(logmix) / TEMPERATURE
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

    # take z from sequence (not dream)
    next_z = sequence_z[i]
    prev_action = sequence_action[i]

    next_restart = 0
    done = False
    if logrestart[0] > 0:
        next_restart = 1
        done = True

    im_pred = (imvae.decode(prev_z.reshape(1, _Z))[0] * 255.).astype(np.uint8)
    im_z_pred_pred = (imvae.decode(prev_z_predicted.reshape(1, _Z))[0] * 255.).astype(np.uint8)
    if True:
        import matplotlib.pyplot as plt

        plt.ion()
        plt.figure("images")
        plt.clf()
        fig, (ax1, ax2) = plt.subplots(
            2, 1, num="images"
        )
        ax1.imshow(im_pred)
        ax2.imshow(im_z_pred_pred)
        # update
        plt.pause(0.01)

    prev_z = next_z
    prev_z_predicted = next_z_predicted
    prev_restart = next_restart
    rnn_state = next_state
