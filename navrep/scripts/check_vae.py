from __future__ import print_function
import numpy as np
import os
from sensor_msgs.msg import LaserScan

from navrep.tools.data_extraction import archive_to_lidar_dataset
from navrep.tools.rings import generate_rings
from navrep.models.vae2d import ConvVAE, reset_graph

DEBUG_PLOTTING = True

# Parameters for training
batch_size = 1
NUM_EPOCH = 100
DATA_DIR = "record"
HOME = os.path.expanduser("~")

vae_model_path = os.path.expanduser("~/navrep/models/V/vae.json")

# create network
reset_graph()
vae = ConvVAE(batch_size=batch_size, is_training=False)

# load
vae.load_json(vae_model_path)

# create training dataset
dataset = archive_to_lidar_dataset("~/navrep/datasets/V/ian", limit=180)
if len(dataset) == 0:
    raise ValueError("no scans found, exiting")
print(len(dataset), "scans in dataset.")

# split into batches:
total_length = len(dataset)
num_batches = int(np.floor(total_length / batch_size))

# rings converter
rings_def = generate_rings(64, 64)

dummy_msg = LaserScan()
dummy_msg.range_max = 100.0
dummy_msg.ranges = range(1080)

for idx in range(num_batches):
    batch = dataset[idx * batch_size : (idx + 1) * batch_size]
    scans = batch
    rings = rings_def["lidar_to_rings"](scans).astype(float)

    obs = rings / rings_def["rings_to_bool"]
    # remove "too close" points
    obs[:, :, 0, :] = 0.0

    rings_pred = vae.encode_decode(obs) * rings_def["rings_to_bool"]
    if True:
        import matplotlib.pyplot as plt

        plt.ion()
        plt.figure("rings")
        plt.clf()
        fig, (ax1, ax2) = plt.subplots(
            2, 1, num="rings",
        )
        ax1.imshow(np.reshape(rings, (64, 64)))
        ax2.imshow(np.reshape(rings_pred, (64, 64)))
        ax1.set_title(idx)
        # update
        plt.pause(0.01)
