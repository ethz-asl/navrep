import os

from navrep.tools.envplayer import EnvPlayer
from navrep.envs.markonedreamenv import MarkOneDreamEnv

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # disable GPU

    denv = MarkOneDreamEnv(temperature=0.)
    player = EnvPlayer(denv)
