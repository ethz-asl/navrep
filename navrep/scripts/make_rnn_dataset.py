from __future__ import print_function
import numpy as np
import os
from pyniel.python_tools.path_tools import make_dir_if_not_exists

from navrep.models.vae2d import ConvVAE, reset_graph
from navrep.tools.rings import generate_rings
from navrep.tools.commonargs import parse_common_args

common_args, _ = parse_common_args()
VARIANT = common_args.environment
print(common_args)
_Z = 32

if VARIANT == "ian":
    SS = 5  # sub-sample dataset to increase DT
    vae_path = os.path.expanduser("~/navrep/models/V/vae.json")
    V_dataset_folder = os.path.expanduser("~/navrep/datasets/V/ian")
    M_dataset_folder = os.path.expanduser("~/navrep/datasets/M/ian")
if VARIANT == "toy":
    SS = None  # no need to sub-sample, toy env is already at DT=0.5
    vae_path = os.path.expanduser("~/navrep/models/V/toyvae.json")
    V_dataset_folder = os.path.expanduser("~/navrep/datasets/V/toy")
    M_dataset_folder = os.path.expanduser("~/navrep/datasets/M/toy")
if VARIANT == "markone":
    SS = None
    vae_path = os.path.expanduser("~/navrep/models/V/markonevae.json")
    V_dataset_folder = os.path.expanduser("~/navrep/datasets/V/markone")
    M_dataset_folder = os.path.expanduser("~/navrep/datasets/M/markone")
if VARIANT == "marktwo":
    SS = None
    vae_path = os.path.expanduser("~/navrep/models/V/marktwovae.json")
    V_dataset_folder = os.path.expanduser("~/navrep/datasets/V/marktwo")
    M_dataset_folder = os.path.expanduser("~/navrep/datasets/M/marktwo")
if VARIANT == "navreptrain":
    SS = None
    vae_path = os.path.expanduser("~/navrep/models/V/navreptrainvae.json")
    V_dataset_folder = os.path.expanduser("~/navrep/datasets/V/navreptrain")
    M_dataset_folder = os.path.expanduser("~/navrep/datasets/M/navreptrain")


# create network
reset_graph()
vae = ConvVAE(z_size=_Z, batch_size=1, is_training=False,)
vae.load_json(vae_path)

# rings converter
rings_def = generate_rings(64, 64)

# labels to learn are x, r, d (obs, reward, done)

files = []
for dirpath, dirnames, filenames in os.walk(V_dataset_folder):
    for filename in [f for f in filenames if f.endswith(".npz")]:
        files.append(os.path.join(dirpath, filename))
files = sorted(files)
for path in files:
    arrays = np.load(path)
    scans = arrays["scans"]
    robotstates = arrays["robotstates"]
    rewards = arrays["rewards"]
    actions = arrays["actions"]
    dones = arrays["dones"]

    if SS is not None:
        scans = np.concatenate([scans[i::SS] for i in range(SS)], axis=0)
        robotstates = np.concatenate([robotstates[i::SS] for i in range(SS)], axis=0)
        rewards = np.concatenate([rewards[i::SS] for i in range(SS)], axis=0)
        actions = np.concatenate([actions[i::SS] for i in range(SS)], axis=0)
        dones_ss = [dones[i::SS] for i in range(SS)]
        for d in dones_ss:
            d[-1] = True
        dones = np.concatenate(dones_ss, axis=0)

    obs = rings_def["lidar_to_rings"](scans).astype(float) / rings_def["rings_to_bool"]

    mus = []
    logvars = []
    batch_size = 100
    for i in range(0, len(obs), batch_size):
        mu, logvar = vae.encode_mu_logvar(obs[i : i + batch_size, :, :, :])
        z = mu + np.exp(logvar / 2.0) * np.random.randn(*logvar.shape)
        mus.extend(mu)
        logvars.extend(logvar)
    mus = np.array(mus).astype(np.float32)
    logvars = np.array(logvars).astype(np.float32)
    savepath = path.replace(V_dataset_folder, M_dataset_folder).replace("scans", "mus_logvars")
    make_dir_if_not_exists(os.path.dirname(savepath))
    np.savez_compressed(
        savepath,
        mus=mus,
        logvars=logvars,
        robotstates=robotstates,
        rewards=rewards,
        actions=actions,
        dones=dones,
    )
    print("z saved to {}".format(savepath))
