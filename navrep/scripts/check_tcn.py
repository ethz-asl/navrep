from __future__ import print_function
import numpy as np
import os

from navrep.tools.rings import generate_rings
from navrep.models.tcn import reset_graph, sample_hps_params, MDNTCN, get_pi_idx
from navrep.models.vae2d import ConvVAE

# parameters
TEMPERATURE = 0.5
_Z = 32

sequence_z_path = os.path.expanduser(
    "~/navrep/datasets/M/ian/000_mus_logvars_robotstates_actions_rewards_dones.npz"
)
rnn_model_path = os.path.expanduser("~/navrep/models/M/tcn.json")
vae_model_path = os.path.expanduser("~/navrep/models/V/vae.json")

reset_graph()
tcn = MDNTCN(sample_hps_params, gpu_mode=False)
vae = ConvVAE(batch_size=1, is_training=False)

vae.load_json(vae_model_path)
tcn.load_json(rnn_model_path)

rings_def = generate_rings(64, 64)

# load sequence image encoding
arrays = np.load(sequence_z_path)
sequence_action = arrays["actions"]
sequence_mu = arrays["mus"]
sequence_logvar = arrays["logvars"]
sequence_restart = arrays["dones"]
sequence_z = sequence_mu + np.exp(sequence_logvar / 2.0) * np.random.randn(
    *(sequence_mu.shape)
)


feed = {
    tcn.input_z: np.reshape(sequence_z[:999], (1, 999, _Z)),
    tcn.input_action: np.reshape(sequence_action[:999], (1, 999, 3)),
    tcn.input_restart: np.reshape(sequence_restart[:999], (1, 999)),
}

[logmix, mean, logstd, logrestart] = tcn.sess.run(
    [tcn.out_logmix, tcn.out_mean, tcn.out_logstd, tcn.out_restart_logits], feed
)

logmix = logmix.reshape((999, _Z, sample_hps_params.num_mixture))
mean = mean.reshape((999, _Z, sample_hps_params.num_mixture))
logstd = logstd.reshape((999, _Z, sample_hps_params.num_mixture))
logrestart = logrestart.reshape((999, 1))

OUTWIDTH = _Z

# adjust temperatures
logmix2 = np.copy(logmix) / TEMPERATURE
logmix2 -= logmix2.max()
logmix2 = np.exp(logmix2)
logmix2 /= logmix2.sum(axis=-1).reshape((999, _Z, 1))

mixture_idx = np.zeros((999, OUTWIDTH))
chosen_mean = np.zeros((999, OUTWIDTH))
chosen_logstd = np.zeros((999, OUTWIDTH))
for i in range(len(mixture_idx)):
    for j in range(OUTWIDTH):
        idx = get_pi_idx(np.random.rand(), logmix2[i, j])
        mixture_idx[i, j] = idx
        chosen_mean[i, j] = mean[i, j][idx]
        chosen_logstd[i, j] = logstd[i, j][idx]

rand_gaussian = np.random.randn(999, OUTWIDTH) * np.sqrt(TEMPERATURE)
next_z_predicted = chosen_mean + np.exp(chosen_logstd) * rand_gaussian
if sample_hps_params.differential_z:
    next_z_predicted = np.reshape(sequence_z[:999], (1, 999, _Z)) + next_z_predicted


rings_pred = vae.decode(sequence_z[:999].reshape(999, _Z)) * rings_def["rings_to_bool"]
rings_z_pred_pred = (
    vae.decode(next_z_predicted.reshape(999, _Z)) * rings_def["rings_to_bool"]
)
predicted_ranges = rings_def["rings_to_lidar"](rings_pred, 1080)

for i in range(len(rings_pred)):
    if True:
        import matplotlib.pyplot as plt

        plt.ion()
        plt.figure("rings")
        plt.clf()
        fig, (ax1, ax2) = plt.subplots(
            2, 1, subplot_kw=dict(projection="polar"), num="rings"
        )
        rings_def["visualize_rings"](rings_pred[i], scan=None, fig=fig, ax=ax1)
        rings_def["visualize_rings"](rings_z_pred_pred[i], scan=None, fig=fig, ax=ax2)
        ax1.set_ylim([0, 10])
        ax1.set_title(i)
        ax2.set_ylim([0, 10])
        # update
        plt.pause(0.01)
