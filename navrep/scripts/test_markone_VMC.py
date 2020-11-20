from navrep.envs.markencodedenv import MarkOneEncodedEnv
from navrep.envs.markenv import FIRST_TRAIN_MAPS, FIRST_TEST_MAPS
from navrep.scripts.test_mark_common import MarkOneCPolicy, markeval
from navrep.tools.commonargs import parse_multiproc_args


if __name__ == "__main__":
    args, _ = parse_multiproc_args()

    policy = MarkOneCPolicy()

    env = MarkOneEncodedEnv(silent=True, maps=FIRST_TRAIN_MAPS)
    S1 = markeval(policy, env,
                  n_episodes=args.n,
                  subset_index=args.subproc_id, n_subsets=args.n_subprocs,
                  render=args.render)

    env = MarkOneEncodedEnv(silent=True, maps=FIRST_TEST_MAPS)
    S2 = markeval(policy, env,
                  n_episodes=args.n,
                  subset_index=args.subproc_id, n_subsets=args.n_subprocs,
                  render=args.render)

    import matplotlib.pyplot as plt
    plt.ioff()
    plt.show()
