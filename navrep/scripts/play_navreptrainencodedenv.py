from navrep.tools.envplayer import EnvPlayer
from navrep.envs.navreptrainencodedenv import NavRepTrainEncodedEnv
from navrep.tools.commonargs import parse_common_args

if __name__ == "__main__":
    args, _ = parse_common_args()

    env = NavRepTrainEncodedEnv(args.backend, args.encoding, scenario='train', gpu=not args.no_gpu)
    player = EnvPlayer(env, render_mode="rings")
