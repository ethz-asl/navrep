import numpy as np
from timeit import default_timer as timer
from tqdm import tqdm

from navrep.envs.navreptrainenv import NavRepTrainEnv
from navrep.tools.commonargs import parse_common_args

if __name__ == "__main__":
    args, _ = parse_common_args()

    n = args.n
    if n is None:
        n = 1000000

    env = NavRepTrainEnv(scenario='train', silent=True, adaptive=False, collect_statistics=False)
    env.reset()

    action = np.array([0., 0., 0.])

    tic = timer()
    for i in tqdm(range(n)):
        env.step(action)
    toc = timer()
    elapsed = toc-tic

    print("Executed {} simulation steps in {:.1f} seconds.".format(n, elapsed))
