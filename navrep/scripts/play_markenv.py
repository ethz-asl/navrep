import os

from navrep.tools.envplayer import EnvPlayer
from navrep.envs.markenv import MarkEnv

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # disable GPU

    env = MarkEnv()
    player = EnvPlayer(env)
