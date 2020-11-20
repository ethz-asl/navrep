import os

from navrep.envs.markencodedenv import MarkTwoEncodedEnv
from navrep.envs.markenv import SECOND_TRAIN_MAPS, SECOND_TEST_MAPS
from navrep.scripts.test_mark_common import MarkTwoCPolicy, markeval

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # disable GPU

    policy = MarkTwoCPolicy()

    env = MarkTwoEncodedEnv(silent=True, maps=SECOND_TRAIN_MAPS)
    S1 = markeval(policy, env, n_episodes=100, render=False)

    env = MarkTwoEncodedEnv(silent=True, maps=SECOND_TEST_MAPS)
    S2 = markeval(policy, env, n_episodes=100, render=False)

    import matplotlib.pyplot as plt
    plt.ioff()
    plt.show()
