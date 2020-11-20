import os

from navrep.envs.markenv import MarkEnv
from navrep.scripts.test_mark_common import load_markeval_statistics, plot_markeval_statistics


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # disable GPU

    # example usage
    env = MarkEnv(silent=True)  # for plotting only
    stats_dir = os.path.expanduser("~/navrep/eval/markeval/markoneencodedenv_2020_Jul_09__10_47_32_PPO_ckpt")
    S = load_markeval_statistics(stats_dir)
    plot_markeval_statistics(S, env)
