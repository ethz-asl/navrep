import os

from navrep.tools.envplayer import EnvPlayer
from navrep.envs.marktwodreamenv import MarkTwoDreamEnv

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # disable GPU

    denv = MarkTwoDreamEnv(temperature=0.)
    player = EnvPlayer(denv)
