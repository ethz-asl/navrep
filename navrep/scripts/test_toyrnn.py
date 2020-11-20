from __future__ import print_function
import numpy as np
import os
import random
import time

from navrep.models.rnn import reset_graph, default_hps, MDNRNN, MAX_GOAL_DIST

_Z = 32

hps = default_hps()

hps = hps._replace(is_training=False)

dataset_folder = os.path.expanduser("~/navrep/datasets/M/toy")
model_hyperparams_path = os.path.expanduser("~/navrep/models/M/toyrnn.hyperparams.pckl")
model_path = os.path.expanduser("~/navrep/models/M/toyrnn.json")
vae_model_path = os.path.expanduser("~/navrep/models/V/toyvae.json")

# load preprocessed data
files = []
for dirpath, dirnames, filenames in os.walk(dataset_folder):
    for filename in [f for f in filenames if f.endswith(".npz")]:
        files.append(os.path.join(dirpath, filename))
all_data = []
for path in files:
    arrays = np.load(path)
    all_data.append(
        [
            arrays["mus"],
            arrays["logvars"],
            arrays["robotstates"],
            arrays["actions"],
            arrays["dones"],
            arrays["rewards"],
        ]
    )
n_total_frames = np.sum([mu.shape[0] for mu, _, _, _, _, _ in all_data])
print("total frames: ", n_total_frames)


reset_graph()
model = MDNRNN(hps)
model.load_json(model_path)

for epoch in range(1):
    epoch_z_costs = []
    epoch_wrongaction_z_costs = []
    #     print('preparing data for epoch', epoch)
    start = time.time()
    # flatten all sequences into one
    mu_sequence = np.zeros((n_total_frames, _Z), dtype=np.float32)
    logvar_sequence = np.zeros((n_total_frames, _Z), dtype=np.float32)
    robotstate_sequence = np.zeros((n_total_frames, 5), dtype=np.float32)
    action_sequence = np.zeros((n_total_frames, 3), dtype=np.float32)
    done_sequence = np.zeros((n_total_frames, 1), dtype=np.float32)
    reward_sequence = np.zeros((n_total_frames, 1), dtype=np.float32)
    i = 0
    random.shuffle(all_data)
    for mu, logvar, robotstate, action, done, reward in all_data:
        L = len(mu)
        mu_sequence[i : i + L, :] = mu.reshape(L, _Z)
        logvar_sequence[i : i + L, :] = logvar.reshape(L, _Z)
        robotstate_sequence[i : i + L, :] = robotstate.reshape(L, 5)
        action_sequence[i : i + L, :] = action.reshape(L, 3)
        done_sequence[i : i + L, :] = done.reshape(L, 1)
        reward_sequence[i : i + L, :] = reward.reshape(L, 1)
        i += L
    # sample z from  mu and logvar
    z_rs_sequence = mu_sequence + np.exp(logvar_sequence / 2.0) * np.random.randn(
        *(mu_sequence.shape)
    )
    # add goalstate (robotstate[:2]) to z
    robotstate_sequence[:, :2] = robotstate_sequence[:, :2] / MAX_GOAL_DIST  # normalize goal dist
    z_rs_sequence = np.concatenate([z_rs_sequence, robotstate_sequence[:, :2]], axis=-1)
    # resize array to be reshapable into sequences and batches
    chunksize = hps.batch_size * hps.max_seq_len  # frames per batch (100'000)
    n_chunks = n_total_frames // chunksize
    # reshape into sequences
    z_rs_sequences = np.reshape(
        z_rs_sequence[: n_chunks * chunksize, :], (-1, hps.max_seq_len, _Z)
    )
    action_sequences = np.reshape(
        action_sequence[: n_chunks * chunksize], (-1, hps.max_seq_len, 3)
    )
    done_sequences = np.reshape(
        done_sequence[: n_chunks * chunksize], (-1, hps.max_seq_len)
    )
    reward_sequences = np.reshape(
        reward_sequence[: n_chunks * chunksize], (-1, hps.max_seq_len)
    )
    num_sequences = len(z_rs_sequences)
    if num_sequences == 0:
        raise ValueError("Not enough data for a single batch")
    # shuffle
    random_idxs = list(range(num_sequences))
    random.shuffle(random_idxs)
    random_idxs = np.reshape(random_idxs, (-1, hps.batch_size))
    # reshape into batches
    z_rs_batches = z_rs_sequences[random_idxs]
    action_batches = action_sequences[random_idxs]
    done_batches = done_sequences[random_idxs]
    reward_batches = reward_sequences[random_idxs]
    num_batches = len(z_rs_batches)
    # result is of size (n_batches, batch_size, seq_len, ...)
    #     print('number of batches', num_batches)
    end = time.time()
    time_taken = end - start
    #     print('time taken to create batches', time_taken)

    batch_state = model.sess.run(model.initial_state)

    for batch_z_rs, batch_action, batch_done, batch_reward in zip(
        z_rs_batches, action_batches, done_batches, reward_batches
    ):

        feed = {
            model.batch_z_rs: batch_z_rs,
            model.batch_action: batch_action,
            model.batch_restart: batch_done,
            model.initial_state: batch_state,
        }
        z_cost = model.sess.run(model.z_cost, feed)
        feed = {
            model.batch_z_rs: batch_z_rs,
            model.batch_action: np.random.random(batch_action.shape),
            model.batch_restart: batch_done,
            model.initial_state: batch_state,
        }
        wrongaction_z_cost = model.sess.run(model.z_cost, feed)
        epoch_z_costs.append(z_cost)
        epoch_wrongaction_z_costs.append(wrongaction_z_cost)

print(np.mean(epoch_z_costs))
print(np.mean(epoch_wrongaction_z_costs))
