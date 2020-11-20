import os

from navrep.tools.envplayer import EnvPlayer
from navrep.envs.navreptrainenv import NavRepTrainEnv

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # disable GPU

    env = NavRepTrainEnv()
    player = EnvPlayer(env)
