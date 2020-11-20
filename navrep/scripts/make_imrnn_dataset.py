from __future__ import print_function
import numpy as np
import os
from navrep.models.vae2d import ConvVAE, reset_graph
from navrep.tools.rings import generate_rings
from pyniel.python_tools.path_tools import make_dir_if_not_exists

# create network
reset_graph()
imvae = ConvVAE(batch_size=1, is_training=True, channels=3,)
imvae.load_json(os.path.expanduser("~/navrep/models/V/imvae.json"))

# rings converter
rings_def = generate_rings(64, 64)

# labels to learn are x, r, d (obs, reward, done)
dataset_folder = os.path.expanduser("~/navrep/datasets/V/im")


files = []
for dirpath, dirnames, filenames in os.walk(dataset_folder):
    for filename in [f for f in filenames if f.endswith(".npz")]:
        files.append(os.path.join(dirpath, filename))
files = sorted(files)
for path in files:
    arrays = np.load(path)
    images = arrays["images"]
    rewards = arrays["rewards"]
    actions = arrays["actions"]
    dones = arrays["dones"]
    obs = images.astype(float) / 255.0

    mus = []
    logvars = []
    for i in range(len(obs)):
        mu, logvar = imvae.encode_mu_logvar(obs[i:i+1, :, :, :])
        z = mu + np.exp(logvar / 2.0) * np.random.randn(*logvar.shape)
        mus.append(mu[0])
        logvars.append(logvar[0])
    mus = np.array(mus).astype(np.float32)
    logvars = np.array(logvars).astype(np.float32)
    savepath = path.replace("imvae", "imrnn").replace("images", "mus_logvars")
    make_dir_if_not_exists(os.path.dirname(savepath))
    np.savez_compressed(
        savepath,
        mus=mus,
        logvars=logvars,
        rewards=rewards,
        actions=actions,
        dones=dones,
    )
    print("z saved to {}".format(savepath))
