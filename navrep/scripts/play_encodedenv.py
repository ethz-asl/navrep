import os

from navrep.tools.envplayer import EnvPlayer
from navrep.envs.encodedenv import EncodedEnv
from navrep.tools.commonargs import parse_common_args

if __name__ == "__main__":
    args, _ = parse_common_args()

    env = EncodedEnv(args.backend, args.encoding)
    player = EnvPlayer(env)

