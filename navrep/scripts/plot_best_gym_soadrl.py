import os
from matplotlib import pyplot as plt

from navrep.tools.commonargs import parse_plotting_args
from plot_gym_training_progress import plot_per_scenario_training_progress

SCENARIO_AXES = {
    "navreptrain" : (0, 0),
    "navrepval" : (1, 0),
}
ERROR_IF_MISSING = False

if __name__ == "__main__":
    args, _ = parse_plotting_args()

    if args.x_axis is None:
        args.x_axis = "train_steps"

    while True:
        basedir = args.logdir
        if basedir is None:
            basedir = "~/best_navrep"
        basedir = os.path.expanduser(basedir)
        best_navrep_dirs = [os.path.join(basedir, o) for o in os.listdir(basedir)
                            if os.path.isdir(os.path.join(basedir,o))]
        plot_per_scenario_training_progress(best_navrep_dirs, SCENARIO_AXES, ERROR_IF_MISSING,
                                            x_axis=args.x_axis, paper_ready=True)
        plt.xlim([-5, 61])
        plt.pause(60)
