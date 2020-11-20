from navrep.tools.envplayer import EnvPlayer
from navrep.envs.archiveenv import ArchiveEnv
from navrep.tools.commonargs import parse_common_args

if __name__ == "__main__":
    args, _ = parse_common_args()

    if args.environment is None:
        args.environment = "irl"

    dataset_folder = "~/navrep/datasets/V/{}".format(args.environment)

    env = ArchiveEnv(dataset_folder)
    player = EnvPlayer(env, step_by_step=True)
