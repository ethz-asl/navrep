import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data.dataloader import DataLoader

from navrep.tools.wdataset import WorldModelDataset
from navrep.models.vae2d import _64
from navrep.models.rnn import MAX_GOAL_DIST

_Z = 32

def log_loss(y_pred, y_true):
    return y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)

def mse(y_pred, y_true):
    return np.mean(np.square(y_pred - y_true))

def rnn_worldmodel_error(rnn, test_dataset_folder, vae):
    sequence_size = rnn.hps.max_seq_len-1
    batch_size = rnn.hps.batch_size
    # load dataset
    seq_loader = WorldModelDataset(test_dataset_folder, sequence_size,
                                   channel_first=False, as_torch_tensors=False, file_limit=64)
    batch_loader = DataLoader(seq_loader, shuffle=False, batch_size=batch_size)
    # iterate over batches
    pbar = tqdm(range(len(batch_loader)*batch_size), total=len(batch_loader)*batch_size).__iter__()
    n_batches = 0
    sum_state_error = 0
    sum_lidar_error = 0
    for x, a, y, x_rs, y_rs, dones in batch_loader:
        # for legacy reasons rnn likes x_rs to be normalized
        x_rs = x_rs / MAX_GOAL_DIST

        # convert img to z
        batch_state = rnn.sess.run(rnn.initial_state)
        # doing the whole batch at once is too mem intensive
        z = np.zeros((batch_size, sequence_size, vae.z_size))  # (b, t, 32)
        for i, single_x in enumerate(x):
            pbar.__next__()
            z[i] = vae.encode(single_x)
        batch_z_rs = np.concatenate([z, x_rs], axis=-1)   # (b, t, 34)
        # add 1 to sequence
        batch_z_rs = np.concatenate([batch_z_rs, np.zeros((batch_size, 1, rnn.hps.seq_width))], axis=1)
        a = np.concatenate([a, np.zeros((batch_size, 1, a.shape[-1]))], axis=1)
        dones = np.concatenate([dones, np.zeros((batch_size, 1))], axis=1)
        # predict
        feed = {
            rnn.batch_z_rs: batch_z_rs,
            rnn.batch_action: a,
            rnn.batch_restart: dones,
            rnn.initial_state: batch_state,
        }
        (logmix, out_mean) = rnn.sess.run([rnn.out_logmix, rnn.out_mean], feed)
        # sample deterministic from mean of MDN
        z_rs_pred = out_mean[(np.arange(len(logmix)),np.argmax(logmix, axis=-1))]
        z_rs_pred = z_rs_pred.reshape(batch_size, sequence_size, rnn.hps.seq_width)
        z_pred = z_rs_pred[:, :, :_Z]
        y_rs_pred = z_rs_pred[:, :, _Z:]
        y_pred_rec = np.zeros((batch_size, sequence_size, _64, _64, 1))
        for i, single_z_pred in enumerate(z_pred):
            y_pred_rec[i] = vae.decode(single_z_pred)

        # cancel the effect of x_rs normalization
        y_rs_pred = y_rs_pred * MAX_GOAL_DIST

        sum_lidar_error += mse(y_pred_rec, y.numpy())  # because binary cross entropy is inf for 0 predictions
        sum_state_error += mse(y_rs_pred, y_rs.numpy())  # mean square error loss
        n_batches += 1
    for _ in pbar:
        pass
    lidar_error = sum_lidar_error / n_batches
    state_error = sum_state_error / n_batches

    return lidar_error, state_error

def vae1d_rnn_worldmodel_error(rnn, test_dataset_folder, vae):
    sequence_size = rnn.hps.max_seq_len-1
    batch_size = rnn.hps.batch_size
    # load dataset
    seq_loader = WorldModelDataset(test_dataset_folder, sequence_size,
                                   lidar_mode="scans",
                                   channel_first=False, as_torch_tensors=False, file_limit=64)
    batch_loader = DataLoader(seq_loader, shuffle=False, batch_size=batch_size)
    # iterate over batches
    pbar = tqdm(range(len(batch_loader)*batch_size), total=len(batch_loader)*batch_size).__iter__()
    n_batches = 0
    sum_state_error = 0
    sum_lidar_error = 0
    for x, a, y, x_rs, y_rs, dones in batch_loader:
        # for legacy reasons rnn likes x_rs to be normalized
        x_rs = x_rs / MAX_GOAL_DIST

        # convert img to z
        batch_state = rnn.sess.run(rnn.initial_state)
        # doing the whole batch at once is too mem intensive
        z = np.zeros((batch_size, sequence_size, vae.z_size))  # (b, t, 32)
        for i, single_x in enumerate(x):
            pbar.__next__()
            z[i] = vae.encode(single_x)
        batch_z_rs = np.concatenate([z, x_rs], axis=-1)   # (b, t, 34)
        # add 1 to sequence
        batch_z_rs = np.concatenate([batch_z_rs, np.zeros((batch_size, 1, rnn.hps.seq_width))], axis=1)
        a = np.concatenate([a, np.zeros((batch_size, 1, a.shape[-1]))], axis=1)
        dones = np.concatenate([dones, np.zeros((batch_size, 1))], axis=1)
        # predict
        feed = {
            rnn.batch_z_rs: batch_z_rs,
            rnn.batch_action: a,
            rnn.batch_restart: dones,
            rnn.initial_state: batch_state,
        }
        (logmix, out_mean) = rnn.sess.run([rnn.out_logmix, rnn.out_mean], feed)
        # sample deterministic from mean of MDN
        z_rs_pred = out_mean[(np.arange(len(logmix)),np.argmax(logmix, axis=-1))]
        z_rs_pred = z_rs_pred.reshape(batch_size, sequence_size, rnn.hps.seq_width)
        z_pred = z_rs_pred[:, :, :_Z]
        y_rs_pred = z_rs_pred[:, :, _Z:]
        y_pred_rec = np.zeros((batch_size, sequence_size, 1080, 1))
        for i, single_z_pred in enumerate(z_pred):
            y_pred_rec[i] = vae.decode(single_z_pred)

        # cancel the effect of x_rs normalization
        y_rs_pred = y_rs_pred * MAX_GOAL_DIST

        sum_lidar_error += mse(y_pred_rec, y.numpy())  # because binary cross entropy is inf for 0 predictions
        sum_state_error += mse(y_rs_pred, y_rs.numpy())  # mean square error loss
        n_batches += 1
    for _ in pbar:
        pass
    lidar_error = sum_lidar_error / n_batches
    state_error = sum_state_error / n_batches

    return lidar_error, state_error

def gpt_worldmodel_error(gpt, test_dataset_folder, device):
    sequence_size = gpt.module.block_size
    batch_size = 128
    # load dataset
    seq_loader = WorldModelDataset(test_dataset_folder, sequence_size,
                                   channel_first=True, as_torch_tensors=True, file_limit=64)
    batch_loader = DataLoader(seq_loader, shuffle=False, batch_size=batch_size)
    # iterate over batches
    batch_loader = tqdm(batch_loader, total=len(batch_loader))
    n_batches = 0
    sum_state_error = 0
    sum_lidar_error = 0
    for x, a, y, x_rs, y_rs, dones in batch_loader:

        # place data on the correct device
        x = x.to(device)
        x_rs = x_rs.to(device)
        a = a.to(device)
        y = y.to(device)
        y_rs = y_rs.to(device)
        dones = dones.to(device)

        y_pred_rec, y_rs_pred, _ = gpt(x, x_rs, a, dones)
        y_pred_rec = y_pred_rec.detach().cpu().numpy()
        y_rs_pred = y_rs_pred.detach().cpu().numpy()

        sum_lidar_error += mse(y_pred_rec, y.cpu().numpy())  # because binary cross entropy is inf for 0
        sum_state_error += mse(y_rs_pred, y_rs.cpu().numpy())  # mean square error loss
        n_batches += 1
    lidar_error = sum_lidar_error / n_batches
    state_error = sum_state_error / n_batches
    return lidar_error, state_error

def gpt1d_worldmodel_error(gpt, test_dataset_folder, device):
    sequence_size = gpt.module.block_size
    batch_size = 128
    # load dataset
    seq_loader = WorldModelDataset(test_dataset_folder, sequence_size,
                                   lidar_mode="scans",
                                   channel_first=True, as_torch_tensors=True, file_limit=64)
    batch_loader = DataLoader(seq_loader, shuffle=False, batch_size=batch_size)
    # iterate over batches
    batch_loader = tqdm(batch_loader, total=len(batch_loader))
    n_batches = 0
    sum_state_error = 0
    sum_lidar_error = 0
    for x, a, y, x_rs, y_rs, dones in batch_loader:

        # place data on the correct device
        x = x.to(device)
        x_rs = x_rs.to(device)
        a = a.to(device)
        y = y.to(device)
        y_rs = y_rs.to(device)
        dones = dones.to(device)

        y_pred_rec, y_rs_pred, _ = gpt(x, x_rs, a, dones)
        y_pred_rec = y_pred_rec.detach().cpu().numpy()
        y_rs_pred = y_rs_pred.detach().cpu().numpy()

        sum_lidar_error += mse(y_pred_rec, y.cpu().numpy())  # because binary cross entropy is inf for 0
        sum_state_error += mse(y_rs_pred, y_rs.cpu().numpy())  # mean square error loss
        n_batches += 1
    lidar_error = sum_lidar_error / n_batches
    state_error = sum_state_error / n_batches
    return lidar_error, state_error

def vae1dlstm_worldmodel_error(vae1dlstm, test_dataset_folder, device):
    sequence_size = 32
    batch_size = 128

    config = vae1dlstm.module.config if hasattr(vae1dlstm, "module") else vae1dlstm.config
    # load dataset
    seq_loader = WorldModelDataset(test_dataset_folder, sequence_size,
                                   lidar_mode="scans",
                                   channel_first=True, as_torch_tensors=True, file_limit=64)
    batch_loader = DataLoader(seq_loader, shuffle=False, batch_size=batch_size)
    # iterate over batches
    batch_loader = tqdm(batch_loader, total=len(batch_loader))
    n_batches = 0
    sum_state_error = 0
    sum_lidar_error = 0
    for x, a, y, x_rs, y_rs, dones in batch_loader:

        # initialize sequence rnn_state
        this_batch_size = x.shape[0]  # last batch can be smaller
        h0 = torch.randn(config.lstm.n_layer, this_batch_size, config.lstm.h_size)
        c0 = torch.randn(config.lstm.n_layer, this_batch_size, config.lstm.h_size)

        # place data on the correct device
        x = x.to(device)
        x_rs = x_rs.to(device)
        a = a.to(device)
        y = y.to(device)
        y_rs = y_rs.to(device)
        dones = dones.to(device)
        h0 = h0.to(device)
        c0 = c0.to(device)

        y_pred_rec, y_rs_pred, _ = vae1dlstm(x, x_rs, a, dones, [h0, c0])
        y_pred_rec = y_pred_rec.detach().cpu().numpy()
        y_rs_pred = y_rs_pred.detach().cpu().numpy()

        sum_lidar_error += mse(y_pred_rec, y.cpu().numpy())  # because binary cross entropy is inf for 0
        sum_state_error += mse(y_rs_pred, y_rs.cpu().numpy())  # mean square error loss
        n_batches += 1
    lidar_error = sum_lidar_error / n_batches
    state_error = sum_state_error / n_batches
    return lidar_error, state_error

def vaelstm_worldmodel_error(vaelstm, test_dataset_folder, device):
    sequence_size = 32
    batch_size = 128

    config = vaelstm.module.config if hasattr(vaelstm, "module") else vaelstm.config
    # load dataset
    seq_loader = WorldModelDataset(test_dataset_folder, sequence_size,
                                   channel_first=True, as_torch_tensors=True, file_limit=64)
    batch_loader = DataLoader(seq_loader, shuffle=False, batch_size=batch_size)
    # iterate over batches
    batch_loader = tqdm(batch_loader, total=len(batch_loader))
    n_batches = 0
    sum_state_error = 0
    sum_lidar_error = 0
    for x, a, y, x_rs, y_rs, dones in batch_loader:

        # initialize sequence rnn_state
        this_batch_size = x.shape[0]  # last batch can be smaller
        h0 = torch.randn(config.lstm.n_layer, this_batch_size, config.lstm.h_size)
        c0 = torch.randn(config.lstm.n_layer, this_batch_size, config.lstm.h_size)

        # place data on the correct device
        x = x.to(device)
        x_rs = x_rs.to(device)
        a = a.to(device)
        y = y.to(device)
        y_rs = y_rs.to(device)
        dones = dones.to(device)
        h0 = h0.to(device)
        c0 = c0.to(device)

        y_pred_rec, y_rs_pred, _ = vaelstm(x, x_rs, a, dones, [h0, c0])
        y_pred_rec = y_pred_rec.detach().cpu().numpy()
        y_rs_pred = y_rs_pred.detach().cpu().numpy()

        sum_lidar_error += mse(y_pred_rec, y.cpu().numpy())  # because binary cross entropy is inf for 0
        sum_state_error += mse(y_rs_pred, y_rs.cpu().numpy())  # mean square error loss
        n_batches += 1
    lidar_error = sum_lidar_error / n_batches
    state_error = sum_state_error / n_batches
    return lidar_error, state_error


if __name__ == "__main__":
    """ Test all worldmodels in logdir """
    import os
    from navrep.tools.commonargs import parse_plotting_args

    from navrep.models.rnn import (reset_graph, default_hps, MDNRNN)
    from navrep.models.vae2d import ConvVAE
    from navrep.models.vae1d import Conv1DVAE
    from navrep.models.gpt import GPT, GPTConfig, load_checkpoint
    from navrep.models.gpt1d import GPT1D
    from navrep.models.vae1dlstm import VAE1DLSTM, VAE1DLSTMConfig
    from navrep.models.vaelstm import VAELSTM, VAELSTMConfig

    _G = 2  # goal dimensions
    _A = 3  # action dimensions
    _RS = 5  # robot state
    _L = 1080  # lidar size
    BLOCK_SIZE = 32  # sequence length (context)

    args, _ = parse_plotting_args()

    gpu = True
    if not gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # disable GPU

    # only implemented for SOADRL
    environment = "navreptrain"
    validation_environment = "navreptrain"
    test_dataset_folder = os.path.expanduser("~/navrep/datasets/V/{}".format(validation_environment))

    UPDATABLE = True

    MODELDIR = args.logdir
    if MODELDIR is None:
        MODELDIR = "~/navrep"
    MODELDIR = os.path.join(MODELDIR, "models")
    MODELDIR = os.path.expanduser(MODELDIR)
    W_MODELDIR = os.path.join(MODELDIR, "W")
    M_MODELDIR = os.path.join(MODELDIR, "M")

    M_models = []
    W_models = []
    try:
        W_models = sorted(os.listdir(W_MODELDIR))[::-1]
    except FileNotFoundError:
        pass
    try:
        M_models = sorted(os.listdir(M_MODELDIR))[::-1]
    except FileNotFoundError:
        pass
    W_dirs = [W_MODELDIR for _ in W_models]
    M_dirs = [M_MODELDIR for _ in M_models]
    for dir_, log in zip(W_dirs+M_dirs, W_models+M_models):
        if log.endswith(".hyperparams.pckl"):
            continue
        path = os.path.join(dir_, log)
        print(log)
        # detect backend type
        if environment+"rnn1d" in log:
            backend = "VAE1D_LSTM"
        elif environment+"rnn" in log:
            backend = "VAE_LSTM"
        elif environment+"gpt1d" in log:
            backend = "GPT1D"
        elif environment+"gpt" in log:
            backend = "GPT"
        elif environment+"vae1dlstm" in log:
            backend = "VAE1DLSTM"
        elif environment+"vaelstm" in log:
            backend = "VAELSTM"
        else:
            print("Backend type for {} not known!".format(log))
            continue

        # Get some param values
        if backend == "GPT":
            from navrep.scripts.train_gpt import _Z, _H
        elif backend == "GPT1D":
            from navrep.scripts.train_gpt1d import _Z, _H
        elif backend == "VAE1DLSTM":
            from navrep.scripts.train_vae1dlstm import _Z, _H
        elif backend == "VAELSTM":
            from navrep.scripts.train_vaelstm import _Z, _H
        elif backend == "VAE_LSTM":
            from navrep.scripts.train_vae import _Z
            from navrep.scripts.train_rnn import _H
        elif backend == "VAE1D_LSTM":
            from navrep.scripts.train_vae1d import _Z
            from navrep.scripts.train_rnn import _H

        # load W / M model
        model = None
        if backend == "VAE_LSTM":
            vae_model_path = os.path.join(MODELDIR,"V", environment+"vae.json")
            reset_graph()
            vae = ConvVAE(z_size=_Z, batch_size=1, is_training=False)
            vae.load_json(vae_model_path)
            hps = default_hps()
            hps = hps._replace(seq_width=_Z+_G, action_width=_A, rnn_size=_H)
            rnn = MDNRNN(hps, gpu_mode=gpu)
            rnn.load_json(path)
        elif backend == "VAE1D_LSTM":
            vae_model_path = os.path.join(MODELDIR,"V", environment+"vae1d.json")
            reset_graph()
            reset_graph()
            vae = Conv1DVAE(z_size=_Z, batch_size=1, is_training=False)
            vae.load_json(vae_model_path)
            hps = default_hps()
            hps = hps._replace(seq_width=_Z+_G, action_width=_A, rnn_size=_H)
            rnn = MDNRNN(hps, gpu_mode=gpu)
            rnn.load_json(path)
        elif backend == "GPT":
            mconf = GPTConfig(BLOCK_SIZE, _H)
            model = GPT(mconf, gpu=gpu)
            load_checkpoint(model, path, gpu=gpu)
            vae = model
            rnn = model
        elif backend == "GPT1D":
            mconf = GPTConfig(BLOCK_SIZE, _H)
            model = GPT1D(mconf, gpu=gpu)
            load_checkpoint(model, path, gpu=gpu)
            vae = model
            rnn = model
        elif backend == "VAELSTM":
            mconf = VAELSTMConfig(_Z, _H)
            model = VAELSTM(mconf, gpu=gpu)
            load_checkpoint(model, path, gpu=gpu)
            vae = model
            rnn = model
        elif backend == "VAE1DLSTM":
            mconf = VAE1DLSTMConfig(_Z, _H)
            model = VAE1DLSTM(mconf, gpu=gpu)
            load_checkpoint(model, path, gpu=gpu)
            vae = model
            rnn = model
        else:
            raise NotImplementedError

        # make torch accept the model if not wrapped in DataParallel
        if model is not None:
            model.module = model

        if backend == "VAE_LSTM":
            lidar_e, state_e = rnn_worldmodel_error(rnn, test_dataset_folder, vae)
        elif backend == "VAE1D_LSTM":
            lidar_e, state_e = vae1d_rnn_worldmodel_error(rnn, test_dataset_folder, vae)
        elif backend == "GPT":
            device = 0
            lidar_e, state_e = gpt_worldmodel_error(rnn, test_dataset_folder, device)
        elif backend == "GPT1D":
            device = 0
            lidar_e, state_e = gpt1d_worldmodel_error(rnn, test_dataset_folder, device)
        elif backend == "VAELSTM":
            device = 0
            lidar_e, state_e = vaelstm_worldmodel_error(rnn, test_dataset_folder, device)
        elif backend == "VAE1DLSTM":
            device = 0
            lidar_e, state_e = vae1dlstm_worldmodel_error(rnn, test_dataset_folder, device)

        print(backend, lidar_e, state_e)

        del rnn
        del vae
        del model
