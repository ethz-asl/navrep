from navrep.tools.envplayer import EnvPlayer
from navrep.envs.toydreamenv import ToyDreamEnv

if __name__ == "__main__":
    denv = ToyDreamEnv(temperature=0.)
    player = EnvPlayer(denv)
