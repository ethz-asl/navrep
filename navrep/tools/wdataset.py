import math
import numpy as np
import os
import torch
from torch.utils.data import Dataset

from navrep.tools.rings import generate_rings
from navrep.envs.navreptrainenv import NavRepTrainEnv
from navrep.scripts.make_vae_dataset import generate_vae_dataset, ORCAPolicy

_64 = 64 # rings size
_CH = 1
_G = 2

LIDAR_NORM_FACTOR = 25.  # maximum typical lidar distance, meters

def scans_to_lidar_obs(scans, lidar_mode, rings_def, channel_first):
    if lidar_mode == "rings":
        N, L = scans.shape
        obs = (
            rings_def["lidar_to_rings"](scans).astype(float)
            / rings_def["rings_to_bool"]
        )
        obs[:, :, 0, :] = 0.0 # remove "too close" points
        # swap channel dimension (compat)
        if channel_first:
            obs = np.moveaxis(obs, -1, 1)
    elif lidar_mode == "scans":
        N, L = scans.shape
        obs = scans / LIDAR_NORM_FACTOR
        obs = obs.reshape(N, L, _CH)  # add single channel
        if channel_first:
            obs = np.moveaxis(obs, -1, 1)
    elif lidar_mode == "images":
        obs = scans / 255.
        if channel_first:
            obs = np.moveaxis(obs, -1, 1)
    else:
        raise NotImplementedError
    return obs

class WorldModelDataset(Dataset):
    def __init__(self, directory, sequence_size,
                 file_limit=None,
                 channel_first=True, as_torch_tensors=True,
                 lidar_mode="rings",
                 pre_convert_obs=False,
                 regen=None):
        """
        Loads data files into a pytorch-compatible dataset

        arguments
        ------
        directory: a path or list of paths in which to look for data files
        sequence_size: the desired length of RNN sequences
        channel_first: if True, outputs samples in (Sequence, Channel, Width, Height) shape,
                        else (Sequence, Width, Height, Channel)
        as_torch_tensors: outputs data samples as torch tensors for convenience
        lidar_mode: "rings", "scans" or "images". Determines how to interpret the sensor data
        pre_convert_obs: converts observation at load time, instead of at sample time
        regen: "navreptrain" or None, in the first case the dataset will be partially replaced with new data
        """
        self.pre_convert_obs = pre_convert_obs
        self.regen = regen
        self.regen_prob = 0.1
        self.lidar_mode = lidar_mode
        self.sequence_size = sequence_size
        self.channel_first = channel_first
        self.as_torch_tensors = as_torch_tensors
        self.rings_def = generate_rings(_64, _64)
        self.data = self._load_data(directory, file_limit=file_limit)
        if self.pre_convert_obs:
            self._preconvert_obs()
        self.regen_head_index = 0  # which part of the data array to regen next
        print("data has %d steps." % (len(self.data["scans"])))

    def _load_data(self, directory, file_limit=None):
        # list all data files
        files = []
        if isinstance(directory, list):
            directories = directory
        elif isinstance(directory, str):
            directories = [directory]
        else:
            raise NotImplementedError
        for dir_ in directories:
            dir_ = os.path.expanduser(dir_)
            for dirpath, dirnames, filenames in os.walk(dir_):
                for filename in [
                    f
                    for f in filenames
                    if f.endswith("scans_robotstates_actions_rewards_dones.npz")
                ]:
                    files.append(os.path.join(dirpath, filename))
        files = sorted(files)
        if file_limit is None:
            file_limit = len(files)
        data = {
            "scans": [],
            "robotstates": [],
            "actions": [],
            "rewards": [],
            "dones": [],
        }
        arrays_dict = {}
        for path in files[:file_limit]:
            arrays_dict = np.load(path)
            for k in np.load(path):
                data[k].append(arrays_dict[k])
        for k in arrays_dict.keys():
            data[k] = np.concatenate(data[k], axis=0)
        return data

    def _preconvert_obs(self):
        print("Pre-converting scans")
        self.data["obs"] = scans_to_lidar_obs(
            self.data["scans"], self.lidar_mode, self.rings_def, self.channel_first)
        print("Done")

    def __len__(self):
        return math.ceil(len(self.data["scans"]) / (self.sequence_size + 1))

    def __getitem__(self, idx):
        if idx >= self.__len__():
            raise IndexError("Index out of bounds")
        if self.regen is not None and idx == len(self)//2:
            if np.random.random() < self.regen_prob:
                print("PARTIAL REGEN")
                self._partial_regen()
        i = idx * self.sequence_size + 1
        scans = self.data["scans"][i : i + self.sequence_size + 1]
        robotstates = self.data["robotstates"][i : i + self.sequence_size + 1]
        actions = self.data["actions"][i : i + self.sequence_size + 1]
        dones = self.data["dones"][i : i + self.sequence_size + 1]
        if self.pre_convert_obs:
            obs = self.data["obs"][i : i + self.sequence_size + 1]
        # robotstates goal only
        robotstates = robotstates[:, :_G]
        # scans as rings
        if not self.pre_convert_obs:
            obs = scans_to_lidar_obs(scans, self.lidar_mode, self.rings_def, self.channel_first)
        # outputs
        x = obs[:-1] * 1.
        y = obs[1:] * 1.
        actions = actions[:-1]
        x_rs = robotstates[:-1]
        y_rs = robotstates[1:]
        dones = dones[:-1]
        # add "black" frames at episode transition AAABBB -> AA__BB, no need to predict A->B
        x[dones == 1] = 0.0
        y[dones == 1] = 0.0
        # torch
        if self.as_torch_tensors:
            x = torch.tensor(x, dtype=torch.float)
            y = torch.tensor(y, dtype=torch.float)
            actions = torch.tensor(actions, dtype=torch.float)
            x_rs = torch.tensor(x_rs, dtype=torch.float)
            y_rs = torch.tensor(y_rs, dtype=torch.float)
            dones = torch.tensor(dones, dtype=torch.float)
        return x, actions, y, x_rs, y_rs, dones

    def _partial_regen(self, n_new_sequences=1):
        if self.regen == "navreptrain":
            env = NavRepTrainEnv(silent=True, scenario='train', adaptive=False)
            env.soadrl_sim.human_num = 20
            data = generate_vae_dataset(
                env, n_sequences=n_new_sequences,
                policy=ORCAPolicy(suicide_if_stuck=True),
                render=False, archive_dir=None)
            if self.pre_convert_obs:
                data["obs"] = scans_to_lidar_obs(
                    data["scans"], self.lidar_mode, self.rings_def, self.channel_first)
        else:
            print("Regen {} failed".format(self.regen))
            return
        for k in self.data.keys():
            N = len(data[k])  # should be the same for each key
            # check end inside loop to avoid having to pick an arbitrary key
            if self.regen_head_index + N > len(self.data[k]):
                self.regen_head_index = 0
            # replace data
            i = self.regen_head_index
            self.data[k][i : i + N] = data[k]
        self.regen_head_index += N
