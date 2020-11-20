from navrep.tools.envplayer import EnvPlayer
from navrep.envs.archiveenv import ArchiveEncodedEnv
from navrep.tools.commonargs import parse_common_args

if __name__ == "__main__":
    args, _ = parse_common_args()

    env = ArchiveEncodedEnv(args.backend, args.encoding, "~/navrep/datasets/V/irl")
    player = EnvPlayer(env, render_mode="rings", step_by_step=True)
