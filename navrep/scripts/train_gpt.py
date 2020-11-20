import os
import matplotlib
if os.path.expandvars("$MACHINE_NAME") in ["leonhard", "euler"]:
    matplotlib.use('agg')
import logging
import os
import math
import time
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from pyniel.python_tools.path_tools import make_dir_if_not_exists

from navrep.models.gpt import GPT, GPTConfig, save_checkpoint, set_seed
from navrep.tools.wdataset import WorldModelDataset
from navrep.tools.test_worldmodel import gpt_worldmodel_error
from navrep.tools.commonargs import parse_common_args

_Z = _H = 64
_S = 32  # sequence length

if __name__ == "__main__":
    args, _ = parse_common_args()

    START_TIME = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")

    if args.environment == "navreptrain":
        dataset_dir = os.path.expanduser("~/navrep/datasets/V/navreptrain")
        data_regen = "navreptrain"
        log_path = os.path.expanduser("~/navrep/logs/W/navreptrain_gpt_train_log_{}.csv".format(START_TIME))
        checkpoint_path = os.path.expanduser("~/navrep/models/W/navreptraingpt")
    else:
        raise NotImplementedError(args.environment)

    if args.dry_run:
        log_path = log_path.replace(os.path.expanduser("~/navrep"), "/tmp/navrep")
        checkpoint_path = checkpoint_path.replace(os.path.expanduser("~/navrep"), "/tmp/navrep")

    make_dir_if_not_exists(os.path.dirname(checkpoint_path))
    make_dir_if_not_exists(os.path.dirname(log_path))
    make_dir_if_not_exists(os.path.expanduser("~/tmp_navrep"))

    # make deterministic
    set_seed(42)

    # set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    mconf = GPTConfig(_S, _H)

    train_dataset = WorldModelDataset(
        dataset_dir, _S,
        pre_convert_obs=True,
        regen=data_regen,
    )

    # training params
    # optimization parameters
    max_steps = args.n
    if max_steps is None:
        max_steps = 222222
    max_epochs = max_steps  # don't stop based on epoch
    batch_size = 128
    learning_rate = 6e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    lr_decay = True  # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    weight_decay = 0.1  # only applied on matmul weights
    warmup_tokens = 512 * 20
    final_tokens = 200 * len(train_dataset) * _S
    num_workers = 0  # for DataLoader

    # create model
    model = GPT(mconf)
    print("GPT trainable params: {}".format(
        sum(p.numel() for p in model.parameters() if p.requires_grad)))

    # take over whatever gpus are on the system
    device = "cpu"
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        model = torch.nn.DataParallel(model).to(device)

    # create the optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    params_decay = [
        p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
    ]
    params_nodecay = [
        p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
    ]
    optim_groups = [
        {"params": params_decay, "weight_decay": weight_decay},
        {"params": params_nodecay, "weight_decay": 0.0},
    ]
    optimizer = optim.AdamW(optim_groups, lr=learning_rate, betas=betas)

    global_step = 0
    tokens = 0  # counter used for learning rate decay
    values_logs = None
    start = time.time()
    for epoch in range(max_epochs):
        is_train = True
        model.train(is_train)
        loader = DataLoader(
            train_dataset,
            shuffle=is_train,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        losses = []
        pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
        for it, (x, a, y, x_rs, y_rs, dones) in pbar:
            global_step += 1

            # place data on the correct device
            x = x.to(device)
            x_rs = x_rs.to(device)
            a = a.to(device)
            y = y.to(device)
            y_rs = y_rs.to(device)
            dones = dones.to(device)

            # forward the model
            with torch.set_grad_enabled(is_train):
                y_pred, y_rs_pred, loss = model(x, x_rs, a, dones, targets=(y, y_rs))
                loss = (
                    loss.mean()
                )  # collapse all losses if they are scattered on multiple gpus
                losses.append(loss.item())

            if is_train:

                # backprop and update the parameters
                model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm_clip)
                optimizer.step()

                # decay the learning rate based on our progress
                if lr_decay:
                    tokens += (
                        a.shape[0] * a.shape[1]
                    )  # number of tokens processed this step
                    if tokens < warmup_tokens:
                        # linear warmup
                        lr_mult = float(tokens) / float(max(1, warmup_tokens))
                    else:
                        # cosine learning rate decay
                        progress = float(tokens - warmup_tokens) / float(
                            max(1, final_tokens - warmup_tokens)
                        )
                        lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                    lr = learning_rate * lr_mult
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = lr
                else:
                    lr = learning_rate

                # report progress
                pbar.set_description(
                    f"epoch {epoch}: train loss {loss.item():.5f}. lr {lr:e}"
                )

                if global_step == 1 or global_step % 1000 == 0:
                    # save plot
                    from matplotlib import pyplot as plt
                    plt.figure("training_status")
                    plt.clf()
                    plt.suptitle("training step {}".format(global_step))
                    f, axes = plt.subplots(3, 5, num="training_status", sharex=True, sharey=True)
                    for i, (ax0, ax1, ax2) in enumerate(axes.T):
                        ax0.imshow(np.squeeze(x.cpu()[0, 5 + i]), cmap=plt.cm.Greys)
                        ax1.imshow(np.squeeze(y.cpu()[0, 5 + i]), cmap=plt.cm.Greys)
                        ax2.imshow(np.squeeze(y_pred.detach().cpu()[0, 5 + i]), cmap=plt.cm.Greys)
                        ax2.set_xlabel("Done {}".format(dones.cpu()[0, 5 + 1]))
                    plt.savefig(os.path.expanduser(
                        "~/tmp_navrep/gpt_step{:07}.png").format(global_step))

        lidar_e = None
        state_e = None
        if epoch % 20 == 0:
            lidar_e, state_e = gpt_worldmodel_error(model, dataset_dir, device)
            save_checkpoint(model, checkpoint_path)

        # log
        end = time.time()
        time_taken = end - start
        start = time.time()
        values_log = pd.DataFrame(
            [[global_step, loss.item(), lidar_e, state_e, time_taken]],
            columns=["step", "cost", "lidar_test_error", "state_test_error", "train_time_taken"],
        )
        if values_logs is None:
            values_logs = values_log.copy()
        else:
            values_logs = values_logs.append(values_log, ignore_index=True)
        if log_path is not None:
            values_logs.to_csv(log_path)

        if not is_train:
            logger.info("test loss: %f", np.mean(losses))

        if global_step >= max_steps:
            break
