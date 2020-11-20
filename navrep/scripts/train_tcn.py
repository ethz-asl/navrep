from __future__ import print_function
from datetime import datetime
import numpy as np
import os
import random
import time
import pickle
from pyniel.python_tools.path_tools import make_dir_if_not_exists
import pandas as pd

from navrep.models.tcn import reset_graph, default_hps, MDNTCN
from navrep.tools.commonargs import parse_common_args

common_args, _ = parse_common_args()
VARIANT = common_args.environment
print(common_args)


hps = default_hps()
# hps.batch_size = 100
# hps.max_seq_len = 1000
N_EPOCHS = 10000
START_TIME = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
_Z = 32

# hps = hps._replace(learning_rate=0.001)

if VARIANT == "ian":
    dataset_folder = os.path.expanduser("~/navrep/datasets/M/ian")
    log_path = os.path.expanduser("~/navrep/logs/M/tcn_train_log_{}.csv".format(START_TIME))
    log_hyperparams_path = os.path.expanduser(
        "~/navrep/logs/M/tcn_train_log_{}.hyperparams.pckl".format(START_TIME))
    model_hyperparams_path = os.path.expanduser("~/navrep/models/M/tcn.hyperparams.pckl")
    model_path = os.path.expanduser("~/navrep/models/M/tcn.json")
    vae_model_path = os.path.expanduser("~/navrep/models/V/vae.json")
if VARIANT == "toy":
    dataset_folder = os.path.expanduser("~/navrep/datasets/M/toy")
    log_path = os.path.expanduser("~/navrep/logs/M/toytcn_train_log_{}.csv".format(START_TIME))
    log_hyperparams_path = os.path.expanduser(
        "~/navrep/logs/M/toytcn_train_log_{}.hyperparams.pckl".format(START_TIME))
    model_hyperparams_path = os.path.expanduser("~/navrep/models/M/toytcn.hyperparams.pckl")
    model_path = os.path.expanduser("~/navrep/models/M/toytcn.json")
    vae_model_path = os.path.expanduser("~/navrep/models/V/toyvae.json")

make_dir_if_not_exists(os.path.dirname(model_path))
make_dir_if_not_exists(os.path.dirname(log_path))

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
            arrays["actions"],
            arrays["dones"],
            arrays["rewards"],
        ]
    )
n_total_frames = np.sum([mu.shape[0] for mu, _, _, _, _ in all_data])
print("total frames: ", n_total_frames)


reset_graph()
model = MDNTCN(hps)

viewer = None
values_logs = None

start = time.time()
for epoch in range(1, N_EPOCHS + 1):
    #     print('preparing data for epoch', epoch)
    epoch_start = time.time()
    # flatten all sequences into one
    mu_sequence = np.zeros((n_total_frames, _Z), dtype=np.float32)
    logvar_sequence = np.zeros((n_total_frames, _Z), dtype=np.float32)
    action_sequence = np.zeros((n_total_frames, 3), dtype=np.float32)
    done_sequence = np.zeros((n_total_frames, 1), dtype=np.float32)
    reward_sequence = np.zeros((n_total_frames, 1), dtype=np.float32)
    i = 0
    random.shuffle(all_data)
    for mu, logvar, action, done, reward in all_data:
        L = len(mu)
        mu_sequence[i : i + L, :] = mu.reshape(L, _Z)
        logvar_sequence[i : i + L, :] = logvar.reshape(L, _Z)
        action_sequence[i : i + L, :] = action.reshape(L, 3)
        done_sequence[i : i + L, :] = done.reshape(L, 1)
        reward_sequence[i : i + L, :] = reward.reshape(L, 1)
        i += L
    # sample z from  mu and logvar
    z_sequence = mu_sequence + np.exp(logvar_sequence / 2.0) * np.random.randn(
        *(mu_sequence.shape)
    )
    # resize array to be reshapable into sequences and batches
    chunksize = hps.batch_size * hps.max_seq_len  # frames per batch (100'000)
    n_chunks = n_total_frames // chunksize
    # reshape into sequences
    z_sequences = np.reshape(
        z_sequence[: n_chunks * chunksize, :], (-1, hps.max_seq_len, _Z)
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
    num_sequences = len(z_sequences)
    # shuffle
    random_idxs = list(range(num_sequences))
    random.shuffle(random_idxs)
    random_idxs = np.reshape(random_idxs, (-1, hps.batch_size))
    # reshape into batches
    z_batches = z_sequences[random_idxs]
    action_batches = action_sequences[random_idxs]
    done_batches = done_sequences[random_idxs]
    reward_batches = reward_sequences[random_idxs]
    num_batches = len(z_batches)
    # result is of size (n_batches, batch_size, seq_len, ...)
    #     print('number of batches', num_batches)
    batch_create_time = time.time() - epoch_start
    #     print('time taken to create batches', batch_create_time)

    for batch_z, batch_action, batch_done, batch_reward in zip(
        z_batches, action_batches, done_batches, reward_batches
    ):

        if False:  # Visually check that the batch is sound
            from navrep.models.vae2d import ConvVAE
            import matplotlib.pyplot as plt
            from navrep.tools.rings import generate_rings

            reset_graph()
            vae = ConvVAE(batch_size=1, is_training=False)
            vae.load_json(vae_model_path)
            rings_def = generate_rings(64, 64)
            rings_pred = vae.decode(batch_z[0]) * rings_def["rings_to_bool"]
            plt.ion()
            for i, ring in enumerate(rings_pred):
                rings_def["visualize_rings"](ring, scan=None)
                plt.ylim([0, 10])
                plt.title(str(batch_action[0, i]))
                plt.pause(0.1)
            exit()
        if False:
            from navrep.models.vae2d import ConvVAE
            from navrep.tools.render import render_lidar_batch
            from navrep.tools.rings import generate_rings

            reset_graph()
            vae = ConvVAE(batch_size=100, is_training=False)
            vae.load_json(vae_model_path)
            rings_def = generate_rings(64, 64)
            batch_decodings = []
            for i in range(batch_z.shape[1]):  # for each sequence step
                rings_pred = vae.decode(batch_z[:, i]) * rings_def["rings_to_bool"]
                predicted_ranges = rings_def["rings_to_lidar"](rings_pred, 1080)
                batch_decodings.append(predicted_ranges)
            for i, predicted_ranges in enumerate(batch_decodings):
                viewer = render_lidar_batch(
                    predicted_ranges, 0, 2 * np.pi, viewer=viewer
                )
                import pyglet

                filename = "/tmp/frame{:03}.png".format(i)
                pyglet.image.get_buffer_manager().get_color_buffer().save(filename)
                print("image file writen : ", filename)

        step = model.sess.run(model.global_step)
        curr_learning_rate = (hps.learning_rate - hps.min_learning_rate) * (
            hps.decay_rate
        ) ** step + hps.min_learning_rate

        feed = {
            model.batch_z: batch_z,
            model.batch_action: batch_action,
            model.batch_restart: batch_done,
            model.lr: curr_learning_rate,
        }

        (train_cost, rmse, z_cost, r_cost, train_step, _) = model.sess.run(
            [
                model.cost,
                model.rmse,
                model.z_cost,
                model.r_cost,
                model.global_step,
                model.train_op,
            ],
            feed,
        )
        if step % 20 == 0 and step > 0:
            end = time.time()
            time_taken = end - start
            start = time.time()
            output_log = (
                "step: %d, lr: %.6f, cost: %.4f, z_cost: %.4f, r_cost: %.4f, train_time_taken: %.4f"
                % (step, curr_learning_rate, train_cost, z_cost, r_cost, time_taken)
            )
            print(output_log)
            model.save_json(model_path)
            with open(model_hyperparams_path, "wb") as f:
                pickle.dump(hps, f)
            values_log = pd.DataFrame(
                [
                    [
                        step,
                        curr_learning_rate,
                        train_cost,
                        rmse,
                        z_cost,
                        r_cost,
                        time_taken,
                    ]
                ],
                columns=[
                    "step",
                    "lr",
                    "cost",
                    "rmse",
                    "z_cost",
                    "r_cost",
                    "train_time_taken",
                ],
            )
            if values_logs is None:
                values_logs = values_log.copy()
            else:
                values_logs = values_logs.append(values_log, ignore_index=True)
            values_logs.to_csv(log_path)
            with open(log_hyperparams_path, "wb") as f:
                pickle.dump(hps, f)

model.save_json(model_path)
