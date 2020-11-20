import os

from navrep.tools.envplayer import EnvPlayer
from navrep.envs.ianenv import IANEnv

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # disable GPU

    env = IANEnv()
    player = EnvPlayer(env)
