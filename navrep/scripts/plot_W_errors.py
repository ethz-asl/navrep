import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
import os
import sys
import numpy as np

from navrep.tools.commonargs import parse_plotting_args
from navrep.scripts.plot_gym_training_progress import backend_colors

MAX_GOAL_DIST = 25.

def get_color(log):
    color = None
    if "gpt1d" in log:
        color = backend_colors["GPT1D"]
    elif "gpt" in log:
        color = backend_colors["GPT"]
    elif "vae1dlstm" in log:
        color = backend_colors["VAE1DLSTM"]
    elif "vaelstm" in log:
        color = backend_colors["VAELSTM"]
    elif "navreptrainrnn1d" in log:
        color = backend_colors["VAE1D_LSTM"]
    elif "navreptrainrnn" in log:
        color = backend_colors["VAE_LSTM"]
    return color


if __name__ == "__main__":
    args, _ = parse_plotting_args()

    UPDATABLE = True
    ANIMATE = False

    LOGDIR = args.logdir
    if LOGDIR is None:
        LOGDIR = "~/navrep"
    LOGDIR = os.path.join(LOGDIR, "logs/W")
    LOGDIR = os.path.expanduser(LOGDIR)
    if len(sys.argv) == 2:
        LOGDIR = sys.argv[1]
    M_LOGDIR = LOGDIR.replace("W", "M")

    plt.close('all')
    i_frame = 0
    while True:
        N_ROWS = 2
        N_COLS = 2
        fig, axes = plt.subplots(N_ROWS, N_COLS, num="Worldmodel Error", sharex=True, sharey=False)
        axes = np.array(axes).reshape((N_ROWS, N_COLS))
        plt.cla()
        logs = []
        M_logs = []
        try:
            logs = sorted(os.listdir(LOGDIR))
        except FileNotFoundError:
            pass
        try:
            M_logs = sorted(os.listdir(M_LOGDIR))
        except FileNotFoundError:
            pass
        dirs = [LOGDIR for _ in logs]
        M_dirs = [M_LOGDIR for _ in M_logs]
        legend = []
        lines = []
        for dir_, log in zip((dirs+M_dirs)[::-1], (logs+M_logs)[::-1]):
            if not log.endswith(".csv"):
                continue
            path = os.path.join(dir_, log)
            # get data
            data = pd.read_csv(path)
            if "lidar_test_error" not in data:
                continue
            x = data["step"].values
            y = data["lidar_test_error"].values  # already normalized in wdataset
            y2 = data["state_test_error"].values / MAX_GOAL_DIST**2  # normalize ourself
    #         y = data["cost"].values
            # filter nans
            valid = np.logical_not(np.isnan(y))
            x = x[valid]
            y = y[valid]
            y2 = y2[valid]
            # plot
            if "1d" in log:
                ax = axes[1, 0]
                ax2 = axes[1, 1]
#                 ax.set_title("Lidar State Prediction Error (MSE) - per ray")
#                 ax2.set_title("Goal-Vel State Prediction Error (MSE)")
                ax.set_title("Lidar State Prediction Error (1D)")
                ax2.set_title("Goal-Vel State Prediction Error")
            else:
                ax = axes[0, 0]
                ax2 = axes[0, 1]
#                 ax.set_title("Worldmodel Prediction Error (MSE) - per rings pixel")
#                 ax2.set_title("Goal-Vel State Prediction Error (MSE)")
                ax.set_title("Lidar State Prediction Error")
                ax2.set_title("Goal-Vel State Prediction Error")
            ax.set_yscale('log')
            ax2.set_yscale('log')
            ax.set_ylabel("MSE Error [unitless]")
            ax.set_xlabel("# of training steps")
            color = get_color(log)
            if ANIMATE:
                yanim = np.ones_like(y) * np.nan
                y2anim = np.ones_like(y2) * np.nan
                line, = ax.plot(x, yanim, color=color, label=log)
                line2, = ax2.plot(x, y2anim, color=line.get_c(), linestyle="--")
                line.set_linewidth(2)
                line2.set_linewidth(2)
                N = 10
                for i in range(0, len(y), N):
                    yanim[i:i+N] = y[i:i+N]
                    y2anim[i:i+N] = y2[i:i+N]
                    line.set_ydata(yanim)
                    line2.set_ydata(y2anim)
                    ax.set_xlim([-1000, 240000])
                    ax2.set_xlim([-1000, 240000])
                    ax.set_ylim([0.005, 0.1])
                    ax2.set_ylim([0.0002, 0.2])
                    plt.pause(0.01)
                    plt.savefig("/tmp/plot_W_error_{:05}.png".format(i_frame))
                    i_frame += 1
                line.set_linewidth(1)
                line2.set_linewidth(1)
            else:
                line, = ax.plot(x, y, color=color, label=log)
                ax2.plot(x, y2, color=line.get_c(), linestyle="--")
            ax.axhline(np.min(y), alpha=0.3, linewidth=1, color=line.get_color())
            lines.append(line)
            legend.append(log)
#         for ax in axes.reshape(-1):
#             ax.legend(bbox_to_anchor=(1.05, 1.))   # quick-search : plot tcn training logs
        plt.ion()
        plt.pause(10.)
