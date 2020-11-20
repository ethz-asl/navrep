import os

from navrep.tools.envplayer import EnvPlayer
from navrep.envs.markencodedenv import MarkOneEncodedEnv
from navrep.envs.markenv import FIRST_TRAIN_MAPS

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # disable GPU

    env = MarkOneEncodedEnv(maps=FIRST_TRAIN_MAPS)
    player = EnvPlayer(env)
