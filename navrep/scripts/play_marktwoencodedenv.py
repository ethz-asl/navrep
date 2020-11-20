import os

from navrep.tools.envplayer import EnvPlayer
from navrep.envs.markencodedenv import MarkTwoEncodedEnv
from navrep.envs.markenv import SECOND_TRAIN_MAPS

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # disable GPU

    env = MarkTwoEncodedEnv(maps=SECOND_TRAIN_MAPS)
    player = EnvPlayer(env)
