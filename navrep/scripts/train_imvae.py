from __future__ import print_function
import numpy as np
import os
from navrep.models.vae2d import ConvVAE, reset_graph
from navrep.tools.data_extraction import rosbag_to_image_dataset

DEBUG_PLOTTING = True

# Parameters for training
batch_size = 100
NUM_EPOCH = 1000  # 10
DATA_DIR = "record"
HOME = os.path.expanduser("~")

model_save_dir = HOME + "/navrep/models/V"
model_save_path = os.path.join(model_save_dir, "imvae.json")
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

# create network
reset_graph()
vae = ConvVAE(batch_size=batch_size, is_training=True, reuse=False, channels=3,)

# create training dataset
dataset, _, _, _ = rosbag_to_image_dataset("~/rosbags/openlab_rosbags/corridor_koze_kids.bag")
if len(dataset) == 0:
    raise ValueError("no images found, exiting")

# split into batches:
total_length = len(dataset)
num_batches = int(np.floor(total_length / batch_size))


# train loop:
print("train", "step", "loss", "recon_loss", "kl_loss")
for epoch in range(NUM_EPOCH):
    np.random.shuffle(dataset)
    for idx in range(num_batches):
        batch = dataset[idx * batch_size : (idx + 1) * batch_size]
        images = batch

        obs = images.astype(float) / 255.0

        feed = {
            vae.x: obs,
        }

        (train_loss, r_loss, kl_loss, train_step, _) = vae.sess.run(
            [vae.loss, vae.r_loss, vae.kl_loss, vae.global_step, vae.train_op], feed
        )

        if (train_step + 1) % 500 == 0:
            if DEBUG_PLOTTING:
                from matplotlib import pyplot as plt

                #           plt.ion()
                plt.figure("training_status")
                plt.clf()
                plt.suptitle("training step {}".format(train_step))
                f, (ax1, ax2) = plt.subplots(2, 1, num="training_status")
                ax1.imshow(images[0])
                ax2.imshow((vae.encode_decode(obs)[0] * 255.0).astype(np.uint8))
                plt.savefig("/tmp/imvae_step{:07}.png".format(train_step))
            #           plt.pause(0.01)
            print("step", (train_step + 1), train_loss, r_loss, kl_loss)
            #     if ((train_step+1) % 5000 == 0):
            vae.save_json(model_save_path)

# finished, final model:
vae.save_json(model_save_path)
