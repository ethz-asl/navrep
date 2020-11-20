from __future__ import print_function
import numpy as np
import os
from sensor_msgs.msg import LaserScan

from navrep.tools.data_extraction import archive_to_lidar_dataset
from navrep.models.siren1d import SIREN1D, reset_graph

DEBUG_PLOTTING = True

# Parameters for training
batch_size = 1
NUM_EPOCH = 100
DATA_DIR = "record"
HOME = os.path.expanduser("~")
MAX_LIDAR_DIST = 25.

siren_model_path = os.path.expanduser("~/navrep/models/V/siren1d.json")

# create network
reset_graph()
siren = SIREN1D(batch_size=1080, is_training=False)

# load
siren.load_json(siren_model_path)

# create training dataset
dataset = archive_to_lidar_dataset("~/navrep/datasets/V/ian", limit=180)
if len(dataset) == 0:
    raise ValueError("no scans found, exiting")
print(len(dataset), "scans in dataset.")
dataset = dataset[:500000:100]

# split into batches:
total_length = len(dataset)
num_batches = int(np.floor(total_length / batch_size))

dummy_msg = LaserScan()
dummy_msg.range_max = 100.0
dummy_msg.ranges = range(1080)

ring_accuracy_per_example = np.ones((len(dataset),)) * -1
for idx in range(num_batches):
    batch = dataset[idx * batch_size : (idx + 1) * batch_size]
    scans = batch

    obs = np.clip(scans.astype(np.float) / MAX_LIDAR_DIST, 0.0, MAX_LIDAR_DIST)
    obs = obs.reshape(batch_size, 1080, 1)

    obs_pred = siren.encode_decode(obs)
    error = abs(obs - obs_pred)
    threshold = np.minimum(0.001, abs(obs * 0.05))
    is_right = error <= threshold
    is_right = np.reshape(is_right, (batch_size, 1080))
    ring_accuracy = np.sum(is_right, axis=1) / 1080.
    ring_accuracy_per_example[idx * batch_size : (idx + 1) * batch_size] = ring_accuracy
    print("{:>4.1f}% - {:>4.3f}".format(
        idx * 100. / num_batches,
        np.mean(ring_accuracy_per_example[ring_accuracy_per_example != -1])
    ), end="\r")
if np.any(ring_accuracy_per_example == -1):
    raise ValueError("something weird.")
dataset_mean_ring_accuracy = np.mean(ring_accuracy_per_example)
dataset_min_ring_accuracy = np.min(ring_accuracy_per_example)
dataset_max_ring_accuracy = np.max(ring_accuracy_per_example)
print(dataset_mean_ring_accuracy)
print(dataset_min_ring_accuracy)
print(dataset_max_ring_accuracy)
