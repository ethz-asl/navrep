import os
from pyniel.python_tools.path_tools import make_dir_if_not_exists

from navrep.tools.commonargs import parse_common_args
from navrep.envs.e2eenv import E2E1DIANEnv
from navrep.scripts.cross_test_navreptrain_in_ianenv import run_test_episodes
from navrep.scripts.test_e2e import E2E1DCPolicy


if __name__ == '__main__':
    args, _ = parse_common_args()

    if args.n is None:
        args.n = 1000
    collect_trajectories = False

    env = E2E1DIANEnv(silent=True, collect_trajectories=collect_trajectories)
    policy = E2E1DCPolicy()

    S = run_test_episodes(env, policy, render=args.render, num_episodes=args.n)

    DIR = os.path.expanduser("~/navrep/eval/crosstest")
    if args.dry_run:
        DIR = "/tmp/navrep/eval/crosstest"
    make_dir_if_not_exists(DIR)
    if collect_trajectories:
        NAME = "e2e1dnavreptrain_in_ianenv_{}.pckl".format(len(S))
        PATH = os.path.join(DIR, NAME)
        S.to_pickle(PATH)
    else:
        NAME = "e2e1dnavreptrain_in_ianenv_{}.csv".format(len(S))
        PATH = os.path.join(DIR, NAME)
        S.to_csv(PATH)
    print("{} written.".format(PATH))
