#!/usr/bin/env python
from navrep.envs.dreamenv import DreamEnv
import os

from navrep.tools.envplayer import EnvPlayer

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # disable GPU

    denv = DreamEnv(temperature=0.)
    player = EnvPlayer(denv)
