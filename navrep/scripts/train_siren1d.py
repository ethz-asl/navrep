from __future__ import print_function
import numpy as np
import os

from navrep.models.siren1d import SIREN1D, reset_graph
from navrep.tools.data_extraction import archive_to_lidar_dataset

DEBUG_PLOTTING = True

# Parameters for training
batch_size = 100
NUM_EPOCH = 10
DATA_DIR = "record"
HOME = os.path.expanduser("~")
MAX_LIDAR_DIST = 25.0

model_save_dir = HOME + "/navrep/models/V"
model_save_path = os.path.join(model_save_dir, "siren1d.json")
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

# create network
reset_graph()
siren = SIREN1D(batch_size=batch_size, is_training=True, reuse=False)

# create training dataset
dataset = archive_to_lidar_dataset(limit=1)
if len(dataset) == 0:
    raise ValueError("no scans found, exiting")
print(len(dataset), "scans in dataset.")
n_scans = len(dataset)

all_obs = np.clip(dataset.astype(np.float) / MAX_LIDAR_DIST, 0.0, MAX_LIDAR_DIST)
train_o = all_obs[:,None,:].reshape(n_scans, 1, 1080, 1)
train_x = np.arange(1080)[None, :].reshape(1, 1080, 1)  # TODO: divide by 1080
train_y = all_obs.reshape((n_scans, 1080, 1))

# batch = sample from (n_scans, n_points)
all_indices = np.array(np.where(all_obs)).T

# split into batches:
num_batches = int(np.floor(len(all_indices) / batch_size))

# train loop:
print("train", "step", "loss", "recon_loss")
for epoch in range(NUM_EPOCH):
    np.random.shuffle(all_indices)
    for idx in range(num_batches):
        batch_indices = all_indices[idx * batch_size : (idx + 1) * batch_size]
        batch_o = train_o[batch_indices[:,0], 0]
        batch_x = train_x[0, batch_indices[:,1]]
        batch_y = train_y[batch_indices[:,0], batch_indices[:,1]]

        feed = {
            siren.x: batch_x,
            siren.o: batch_o,
            siren.y_true: batch_y,
        }

        (train_loss, r_loss, train_step, y_pred, _) = siren.sess.run(
            [siren.loss, siren.r_loss, siren.global_step, siren.y, siren.train_op], feed
        )

        print("{:>4.1f}%".format(100. * (train_step % 500) / 500,), end="\r")
        if (train_step + 1) % 500 == 0:
            print("step", (train_step + 1), train_loss, r_loss)
        if (train_step + 1) % 5000 == 0:
            siren.save_json(model_save_path)
            if DEBUG_PLOTTING:
                obs = batch_o[0]
                full_y = np.zeros((1080,))
                for i in range(0, 1000, batch_size):
                    print("{:>4.1f}%".format(i / 10.,), end="\r")
                    batch_y = obs[i:i+batch_size]
                    batch_x = np.arange(i, i+100).reshape((batch_size, 1))
                    batch_o = np.repeat(obs[None,:], batch_size, axis=0)
                    feed = {
                        siren.x: batch_x,
                        siren.o: batch_o,
                        siren.y_true: batch_y,
                    }
                    y_pred = siren.sess.run(siren.y, feed)
                    full_y[i:i+batch_size] = np.squeeze(y_pred)
                from matplotlib import pyplot as plt

                plt.ion()
                plt.cla()
                plt.plot(obs[:,0])
                plt.plot(full_y)
                plt.title("epoch #{}".format(epoch))
                plt.pause(0.01)

print("Done.")

# finished, final model:
siren.save_json(model_save_path)
