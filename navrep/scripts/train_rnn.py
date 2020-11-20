from __future__ import print_function
import numpy as np
import os
from datetime import datetime
import random
import time
import pickle
from pyniel.python_tools.path_tools import make_dir_if_not_exists
import pandas as pd

from navrep.models.vae2d import ConvVAE
from navrep.models.vae1d import Conv1DVAE
from navrep.models.rnn import reset_graph, default_hps, MDNRNN, MAX_GOAL_DIST
from navrep.tools.test_worldmodel import rnn_worldmodel_error, vae1d_rnn_worldmodel_error
from navrep.tools.commonargs import parse_common_args
from navrep.scripts.train_vae import _Z

_H = 512
_G = 2  # goal states
_A = 3  # action dims

if __name__ == "__main__":
    common_args, _ = parse_common_args()
    VARIANT = common_args.environment

    START_TIME = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    MAX_STEPS = common_args.n
    if MAX_STEPS is None:
        MAX_STEPS = 222222
    N_EPOCHS = MAX_STEPS  # don't limit based on epoch
    VAE_TYPE = "1d" if common_args.backend == "VAE1D_LSTM" else ""

    hps = default_hps()
    hps = hps._replace(seq_width=_Z+_G, action_width=_A, rnn_size=_H)
    print(hps)
    # hps.batch_size = 100
    # hps.max_seq_len = 1000
    # hps = hps._replace(learning_rate=0.0001)

    if VARIANT == "ian":
        dataset_folder = os.path.expanduser("~/navrep/datasets/M/ian")
        test_dataset_folder = os.path.expanduser("~/navrep/datasets/V/ian")
        log_path = os.path.expanduser("~/navrep/logs/M/rnn_train_log_{}.csv".format(START_TIME))
        log_hyperparams_path = os.path.expanduser(
            "~/navrep/logs/M/rnn_train_log_{}.hyperparams.pckl".format(START_TIME))
        model_hyperparams_path = os.path.expanduser("~/navrep/models/M/rnn.hyperparams.pckl")
        model_path = os.path.expanduser("~/navrep/models/M/rnn.json")
        vae_model_path = os.path.expanduser("~/navrep/models/V/vae.json")
    if VARIANT == "toy":
        dataset_folder = os.path.expanduser("~/navrep/datasets/M/toy")
        test_dataset_folder = os.path.expanduser("~/navrep/datasets/V/toy")
        log_path = os.path.expanduser("~/navrep/logs/M/toyrnn_train_log_{}.csv".format(START_TIME))
        log_hyperparams_path = os.path.expanduser(
            "~/navrep/logs/M/toyrnn_train_log_{}.hyperparams.pckl".format(START_TIME))
        model_hyperparams_path = os.path.expanduser("~/navrep/models/M/toyrnn.hyperparams.pckl")
        model_path = os.path.expanduser("~/navrep/models/M/toyrnn.json")
        vae_model_path = os.path.expanduser("~/navrep/models/V/toyvae.json")
    if VARIANT == "markone":
        dataset_folder = os.path.expanduser("~/navrep/datasets/M/markone")
        test_dataset_folder = os.path.expanduser("~/navrep/datasets/V/markone")
        log_path = os.path.expanduser("~/navrep/logs/M/markonernn_train_log_{}.csv".format(START_TIME))
        log_hyperparams_path = os.path.expanduser(
            "~/navrep/logs/M/markonernn_train_log_{}.hyperparams.pckl".format(START_TIME))
        model_hyperparams_path = os.path.expanduser("~/navrep/models/M/markonernn.hyperparams.pckl")
        model_path = os.path.expanduser("~/navrep/models/M/markonernn.json")
        vae_model_path = os.path.expanduser("~/navrep/models/V/markonevae.json")
    if VARIANT == "marktwo":
        dataset_folder = os.path.expanduser("~/navrep/datasets/M/marktwo")
        test_dataset_folder = os.path.expanduser("~/navrep/datasets/V/marktwo")
        log_path = os.path.expanduser("~/navrep/logs/M/marktwornn_train_log_{}.csv".format(START_TIME))
        log_hyperparams_path = os.path.expanduser(
            "~/navrep/logs/M/marktwornn_train_log_{}.hyperparams.pckl".format(START_TIME))
        model_hyperparams_path = os.path.expanduser("~/navrep/models/M/marktwornn.hyperparams.pckl")
        model_path = os.path.expanduser("~/navrep/models/M/marktwornn.json")
        vae_model_path = os.path.expanduser("~/navrep/models/V/marktwovae.json")
    if VARIANT == "navreptrain":
        dataset_folder = os.path.expanduser("~/navrep/datasets/M/navreptrain")
        test_dataset_folder = os.path.expanduser("~/navrep/datasets/V/navreptrain")
        log_path = os.path.expanduser(
            "~/navrep/logs/M/navreptrainrnn{}_train_log_{}.csv".format(VAE_TYPE, START_TIME))
        log_hyperparams_path = os.path.expanduser(
            "~/navrep/logs/M/navreptrainrnn{}_train_log_{}.hyperparams.pckl".format(VAE_TYPE, START_TIME))
        model_hyperparams_path = os.path.expanduser(
            "~/navrep/models/M/navreptrainrnn{}.hyperparams.pckl".format(VAE_TYPE))
        model_path = os.path.expanduser("~/navrep/models/M/navreptrainrnn{}.json".format(VAE_TYPE))
        vae_model_path = os.path.expanduser("~/navrep/models/V/navreptrainvae{}.json".format(VAE_TYPE))

    if common_args.dry_run:
        log_path = log_path.replace(os.path.expanduser("~/navrep"), "/tmp/navrep")
        log_hyperparams_path = log_hyperparams_path.replace(os.path.expanduser("~/navrep"), "/tmp/navrep")
        model_path = model_path.replace(os.path.expanduser("~/navrep"), "/tmp/navrep")
        model_hyperparams_path = model_hyperparams_path.replace(os.path.expanduser("~/navrep"), "/tmp/navrep")

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
                arrays["robotstates"],
                arrays["actions"],
                arrays["dones"],
                arrays["rewards"],
            ]
        )
    n_total_frames = np.sum([mu.shape[0] for mu, _, _, _, _, _ in all_data])
    chunksize = hps.batch_size * hps.max_seq_len  # frames per batch (100'000)
    print("total frames: ", n_total_frames)
    if n_total_frames < chunksize:
        raise ValueError()

    reset_graph()
    model = MDNRNN(hps)
    model.print_trainable_params()
    vae = None

    viewer = None
    values_logs = None

    start = time.time()
    for epoch in range(1, N_EPOCHS + 1):
        #     print('preparing data for epoch', epoch)
        batches_start = time.time()
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
        z_sequence = mu_sequence + np.exp(logvar_sequence / 2.0) * np.random.randn(
            *(mu_sequence.shape)
        )
        # add goalstate (robotstate[:2]) to z
        robotstate_sequence[:, :_G] = robotstate_sequence[:, :_G] / MAX_GOAL_DIST  # normalize goal dist
        z_rs_sequence = np.concatenate([z_sequence, robotstate_sequence[:, :_G]], axis=-1)
        # resize array to be reshapable into sequences and batches
        n_chunks = n_total_frames // chunksize
        # reshape into sequences
        z_rs_sequences = np.reshape(
            z_rs_sequence[: n_chunks * chunksize, :], (-1, hps.max_seq_len, _Z+_G)
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
        batches_end = time.time()
        batch_time_taken = batches_end - batches_start
        #     print('time taken to create batches', batch_time_taken)

        batch_state = model.sess.run(model.initial_state)

        for batch_z_rs, batch_action, batch_done, batch_reward in zip(
            z_rs_batches, action_batches, done_batches, reward_batches
        ):
            step = model.sess.run(model.global_step)
            curr_learning_rate = (hps.learning_rate - hps.min_learning_rate) * (
                hps.decay_rate
            ) ** step + hps.min_learning_rate

            feed = {
                model.batch_z_rs: batch_z_rs,
                model.batch_action: batch_action,
                model.batch_restart: batch_done,
                model.initial_state: batch_state,
                model.lr: curr_learning_rate,
            }

            (train_cost, z_cost, r_cost, batch_state, train_step, _) = model.sess.run(
                [
                    model.cost,
                    model.z_cost,
                    model.r_cost,
                    model.final_state,
                    model.global_step,
                    model.train_op,
                ],
                feed,
            )

            lidar_e = None
            state_e = None
            if step % 200 == 0:
                # load VAE
                if VAE_TYPE == "1d":
                    if vae is None:
                        vae = Conv1DVAE(z_size=_Z, batch_size=model.hps.max_seq_len-1, is_training=False)
                        vae.load_json(vae_model_path)
                    lidar_e, state_e = vae1d_rnn_worldmodel_error(model, test_dataset_folder, vae)
                else:
                    if vae is None:
                        vae = ConvVAE(z_size=_Z, batch_size=model.hps.max_seq_len-1, is_training=False)
                        vae.load_json(vae_model_path)
                    lidar_e, state_e = rnn_worldmodel_error(model, test_dataset_folder, vae)

                print("Test: lidar error {}, state error {}".format(lidar_e, state_e))
                model.save_json(model_path)

            if step % 20 == 0 and step > 0:
                end = time.time()
                time_taken = end - start
                start = time.time()
                output_log = (
                    "step: %d, lr: %.6f, cost: %.4f, z_cost: %.4f, r_cost: %.4f, train_time_taken: %.4f"
                    % (step, curr_learning_rate, train_cost, z_cost, r_cost, time_taken)
                )
                print(output_log)
                # log
                values_log = pd.DataFrame(
                    [[step, curr_learning_rate, train_cost, z_cost, r_cost, time_taken, lidar_e, state_e]],
                    columns=["step", "lr", "cost", "z_cost", "r_cost", "train_time_taken",
                             "lidar_test_error", "state_test_error"],
                )
                if values_logs is None:
                    values_logs = values_log.copy()
                else:
                    values_logs = values_logs.append(values_log, ignore_index=True)
                values_logs.to_csv(log_path)
                with open(log_hyperparams_path, "wb") as f:
                    pickle.dump(hps, f)

            if common_args.render:  # Visually check that the batch is sound
                import matplotlib.pyplot as plt
                from navrep.tools.rings import generate_rings

                reset_graph()
                vae = ConvVAE(z_size=_Z, batch_size=1, is_training=False)
                vae.load_json(vae_model_path)
                rings_def = generate_rings(64, 64)
                rings_pred = vae.decode(batch_z_rs[0, :, :_Z]) * rings_def["rings_to_bool"]
                plt.ion()
                for i, ring in enumerate(rings_pred):
                    rings_def["visualize_rings"](ring, scan=None)
                    plt.scatter(batch_z_rs[0, i, _Z], batch_z_rs[0, i, 33], color='red')
                    plt.ylim([0, 10])
                    plt.title("{:.1f} {:.1f} {:.1f}".format(*batch_action[0, i]))
                    plt.pause(0.5)
                exit()
            if False:  # render all sequences in batch at once
                from navrep.tools.render import render_lidar_batch
                from navrep.tools.rings import generate_rings

                reset_graph()
                vae = ConvVAE(z_size=_Z, batch_size=100, is_training=False)
                vae.load_json(vae_model_path)
                rings_def = generate_rings(64, 64)
                batch_decodings = []
                for i in range(batch_z_rs.shape[1]):  # for each sequence step
                    rings_pred = vae.decode(batch_z_rs[:, i, :_Z]) * rings_def["rings_to_bool"]
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

            if MAX_STEPS is not None:
                if train_step > MAX_STEPS:
                    break
        if MAX_STEPS is not None:
            if train_step > MAX_STEPS:
                break

    model.save_json(model_path)
