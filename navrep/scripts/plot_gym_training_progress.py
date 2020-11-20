import os
import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import dates as md
import matplotlib
from rich.console import Console
from rich.table import Table
from datetime import timedelta

from navrep.tools.commonargs import parse_plotting_args

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

MILLION = 1000000
_N = 66


def moving_average(x, w):
    mva = np.convolve(x, np.ones(w), 'valid') / w
    if len(x) > len(mva):
        mva = np.pad(mva, (len(x) - len(mva), 0))
    if len(mva) > len(x):
        mva = mva[:len(x)]
    return mva

def smooth(x, weight):
    """ Weight between 0 and 1 """
    last = x[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in x:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
    return smoothed


SCENARIO_AXES = {
    "irosasl1"         : (0, 0),
    "irosasl2"         : (0, 1),
    "irosasl3"         : (0, 2),
    "irosasl4"         : (0, 3),
    "irosasl5"         : (0, 4),
    "irosasl6"         : (0, 5),
    "irosasl_office_j1": (1, 0),
    "irosasl_office_j2": (1, 1),
    "irosasl_office_j3": (1, 2),
    "irosasl_office_j4": (1, 3),
    "irosasl_office_j5": (1, 4),
    "irosasl_office_j6": (1, 5),
    "irosunity_scene_map1": (2, 0),
    "irosunity_scene_map2": (2, 1),
    "irosunity_scene_map3": (2, 2),
    "irosunity_scene_map4": (2, 3),
    "irosunity_scene_map5": (2, 4),
    "irosunity_scene_map6": (2, 5),
    "rlasl1"           : (3, 0),
    "rlasl2"           : (3, 1),
    "rlasl3"           : (3, 2),
    "rlasl_office_j1"  : (4, 0),
    "rlasl_office_j2"  : (4, 1),
    "rlasl_office_j3"  : (4, 2),
    "rlunity_scene_map1"  : (5, 0),
    "rlunity_scene_map2"  : (5, 1),
    "rlunity_scene_map3"  : (5, 2),
    "marktrain"        : (6, 0),
    "marksimple"       : (7, 0),
    "markcomplex"      : (7, 1),
    "marktm1"          : (7, 2),
    "marktm2"          : (7, 3),
    "marktm3"          : (7, 4),
}
ERROR_IF_MISSING = True

def parse_logfiles(navrep_dirs):
    best_navrep_names = [os.path.basename(path) for path in navrep_dirs]

    all_logpaths = []
    all_parents = []
    for name, dir_ in zip(best_navrep_names, navrep_dirs):
        logdir = os.path.join(dir_, "logs/gym")
        try:
            logfiles = sorted([file for file in os.listdir(logdir) if ".csv" in file])
        except FileNotFoundError:
            logfiles = []
        logpaths = [os.path.join(logdir, logfile) for logfile in logfiles]
        logparents = [name for _ in logfiles]
        all_logpaths.extend(logpaths)
        all_parents.extend(logparents)
    return all_logpaths, all_parents

def plot_per_scenario_training_progress(logdirs, scenario_axes, error_if_missing,
                                        x_axis="train_steps", paper_ready=False):
    SHOW_DIFFICULTY = True
    ANIMATE = False
    logpaths, parents = parse_logfiles(logdirs)

    # get set of all scenarios in all logpaths
    all_scenarios = []
    for logpath in logpaths:
        S = pd.read_csv(logpath)
        scenarios = sorted(list(set(S["scenario"].values)))
        all_scenarios.extend(scenarios)
    all_scenarios = sorted(list(set(all_scenarios)))

    print()
    print("Plotting scenario rewards")
    print()
    plt.ion()
    plt.figure("scenario rewards")
    plt.clf()
    N_ROWS, N_COLS = np.max(np.array(list(scenario_axes.values())), axis=0) + 1
    fig, axes = plt.subplots(N_ROWS, N_COLS, num="scenario rewards", sharex=True, sharey=True)
    axes = np.array(axes).reshape((N_ROWS, N_COLS))
    fig.suptitle("Per-Scenario Training Progress")
    # delete empty axes
    # not_present = [scenario for scenario in scenario_axes.keys() if scenario not in all_scenarios]
    # for scenario in not_present:
    #     ax_i, ax_j = scenario_axes[scenario]
    #     ax = axes[ax_i][ax_j]
    #     fig.delaxes(ax)

    console = Console()
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Parent", style="dim")
    table.add_column("Name")
    table.add_column("Steps", justify="right")
    table.add_column("Reward", justify="right")

    lines = []
    legends = []
    total_train_time = 0.
    i_frame = 0
    for logpath, parent in zip(logpaths, parents):
        logname = os.path.basename(logpath)
        line = None
        color, style = get_color_and_style(logpath)
        S = pd.read_csv(logpath)
        add_train_steps_to_statistics(S)
        for scenario in all_scenarios:
            try:
                ax_i, ax_j = scenario_axes[scenario]
            except KeyError:
                if error_if_missing:
                    raise
                else:
                    continue
            ax = axes[ax_i][ax_j]
            scenario_S = S[S["scenario"] == scenario]
            if x_axis == "train_steps":
                x = scenario_S["train_steps"].values
                x = x / MILLION
            elif x_axis == "total_steps":
                x = scenario_S["total_steps"].values
                x = x / MILLION
            elif x_axis == "wall_time":
                try:
                    x = scenario_S["wall_time"].values
                except KeyError:
                    print("{} has no wall_time info".format(logpath))
                    continue
                x = md.epoch2num(x)
            else:
                raise NotImplementedError
            if len(x) == 0:
                continue
            ylabel = "reward"
            n = np.max(scenario_S["train_steps"].values) / MILLION
            rewards = scenario_S["reward"].values
#             x = np.concatenate([[0], x])
#             rewards = np.concatenate([[0], rewards])
            smooth_rewards = smooth(rewards, 0.99)
            if scenario == "navreptrain":
                y = smooth_rewards
                linestyle = "solid"
                # y as progress rate
                if True:
                    if x_axis == "wall_time":
                        y = scenario_S["train_steps"].values / MILLION
                        SHOW_DIFFICULTY = False
                        linestyle = style
                        ylabel = "Million train steps"
                        # predicted end
                        if n < _N:
                            predicted_end = (x[-1] - x[0]) * 1. * _N / n + x[0]
                            ax.plot([x[-1], predicted_end], [n, _N],
                                    linestyle=linestyle, color=color, linewidth=1., alpha=0.2)
                # plot main reward line
                tmp, = ax.plot(x, y, linewidth=1, linestyle=linestyle, color=color)
                color = tmp.get_c()
                # add line for difficulty
                if SHOW_DIFFICULTY:
#                 ax.plot(x, np.concatenate([[0], scenario_S["num_walls"].values]),
#                         linestyle="--", color=color, linewidth=1)
                    ax.plot(x, scenario_S["num_walls"].values,
                            linestyle="--", color=color, linewidth=1)
                # did the training stall?
                if x_axis == "wall_time":
                    if n < _N:
                        last = scenario_S["wall_time"].values[-1]
                        elapsed_hours_since_last_update = (time.time() - last) / 3600.
                        if elapsed_hours_since_last_update >= 6:
                            print("WARNING: {} may be stalled "
                                  "(incomplete and last update {} hours ago)".format(
                                      logpath, timedelta(hours=elapsed_hours_since_last_update)))
            elif scenario == "navrepval":
                y = rewards
                ylabel = "success rate [%]"
                # plot main reward line
                if ANIMATE:
                    yanim = np.ones_like(y) * np.nan
                    line, = ax.plot(x, yanim, linestyle=style, linewidth=1, color=color)
                    line.set_linewidth(2)
                    if x[-1] < 60:
                        continue
                    for i in range(len(y)):
                        yanim[i] = y[i]
                        line.set_ydata(yanim)
                        ax.set_xlim([-10, 66])
                        ax.set_ylim([-2, 100])
                        plt.pause(0.01)
                        plt.savefig("/tmp/plot_best_gym_{:05}.png".format(i_frame))
                        i_frame += 1
                        if x[i] > 66:
                            break
                    line.set_linewidth(1)
                else:
                    line, = ax.plot(x, y, linestyle=style, linewidth=1, color=color)
                color = line.get_c()
                top = np.max(y)
                # add best hline
                ax.axhline(top, color=color, linestyle=style, linewidth=1, alpha=0.5)
                # circle best score
                if not paper_ready:
                    ax.scatter(x[np.argmax(y)], top, marker='o', facecolor="none", edgecolor=color)
                # predicted end
                if x_axis == "wall_time":
                    train_time = x[-1] - x[0]
                    total_train_time += train_time
                    if n < _N:
                        predicted_end = train_time * _N * 1. / n + x[0]
                        ax.plot([x[-1], predicted_end], [y[-1], y[-1]],
                                linestyle=style, color=color, linewidth=1., alpha=0.2)
                        ax.scatter(predicted_end, y[-1], marker='x', facecolor=color)
                # table
                table.add_row(
                    parent,
                    logname,
                    "{:.1f}".format(n) if n > _N else "[dim]{:.1f}[/dim]".format(n),
                    "{:.1f}".format(top)
                )
            else:
                y = rewards
                # plot main reward line
                line, = ax.plot(x, smooth_rewards, linewidth=1, color=color)
                color = line.get_c()
                # add episode reward scatter
                ax.plot(x, y, color=color, marker=',', linewidth=0, label=scenario)
            # add vertical line at end of finished runs
            if x_axis == "wall_time":
                if n > _N:
                    ax.axvline(x[-1], linestyle=style, linewidth=1, color=color)
                else:
                    ax.scatter(x[-1], y[-1], marker='>', facecolor="none", edgecolor=color)
            ax.set_ylim([-1,101])
            ax.set_ylabel(ylabel)
            if x_axis == "train_steps":
                ax.set_xlabel("Million Train Steps")
            elif x_axis == "total_steps":
                ax.set_xlabel("Million Steps (Eval env - Train env is ~8x larger)")
            elif x_axis == "wall_time":
                ax.set_xlabel("Wall Time")
                xfmt = md.DateFormatter('%d-%b-%Y %H:%M:%S')
                ax.xaxis.set_major_formatter(xfmt)
                from matplotlib.ticker import MultipleLocator
                ax.xaxis.set_minor_locator(MultipleLocator(1))
                ax.xaxis.set_major_locator(MultipleLocator(3))
                ax.grid(which='minor', axis='x', linestyle='-')
                ax.grid(which='major', axis='x', linestyle='-')
            ax.set_title(scenario)
        if line is not None:
            lines.append(line)
            legends.append(parent + ": " + logname)
    # add lucia best hline
    try:
        ax_i, ax_j = scenario_axes["navrepval"]
    except KeyError:
        pass
    ax = axes[ax_i][ax_j]
    line = ax.axhline(85., color="red", linewidth=1, alpha=0.5)
    lines.append(line)
    legends.append("soadrl")
    # add current time
    if x_axis == "wall_time":
        print("Total train time: {} days".format(total_train_time))
        try:
            ax_i, ax_j = scenario_axes["navrepval"]
        except KeyError:
            pass
        ax = axes[ax_i][ax_j]
        ax.axvline(md.epoch2num(time.time()), color='k', linewidth=1)
        try:
            ax_i, ax_j = scenario_axes["navreptrain"]
        except KeyError:
            pass
        ax = axes[ax_i][ax_j]
        ax.axvline(md.epoch2num(time.time()), color='k', linewidth=1)

    console.print(table)
    if not paper_ready:
        fig.legend(lines, legends, bbox_to_anchor=(1.05, 1.))

def add_train_steps_to_statistics(S):
    # here we assume 10000 env.steps happened for every 20 episodes of eval env
    # but with 6 envs, that's 60000 training steps
    is_eval_scenario = S["scenario"] == "navreptrain"
    is_20th_eval_scenario = np.mod(np.cumsum(is_eval_scenario), 20) == 0
    train_steps = np.cumsum(is_20th_eval_scenario * 60000)
    S["train_steps"] = train_steps


backend_colors = {
    "VAE1D_LSTM": "lightskyblue",
    "VAE_LSTM": "cornflowerblue",
    "VAE1DLSTM": "mediumaquamarine",
    "VAELSTM": "mediumseagreen",
    "GPT1D": "khaki",
    "GPT": "gold",
    "E2E1D": "lightgrey",
    "E2E": "grey",
}
envname_colors = {
    "e2e1dnavreptrain": "lightgrey",
    "e2enavreptrain": "grey",
    "lucianavreptrain": "red",
}
encoding_styles = {
    "V_ONLY": "dotted",
    "M_ONLY": "dashed",
    "VM": "solid"
}

envname_labels = {
    "e2e1dnavreptrain": "End-to-end (1D)",
    "e2enavreptrain": "End-to-end",
    "lucianavreptrain": "SOADRL",
}
backend_labels = {
    "VAE1D_LSTM": "Modular (1D)",
    "VAE_LSTM": "Modular",
    "VAE1DLSTM": "Joint (1D)",
    "VAELSTM": "Joint",
    "GPT1D": "Transformer (1D)",
    "GPT": "Transformer",
    "E2E1D": "End-to-end (1D)",
    "E2E": "End-to-end",
}
encoding_labels = {
    "V_ONLY": "z only",
    "M_ONLY": "h only",
    "VM": "z+h"
}


def get_color_and_style(logpath):
    backend, encoding = get_backend_and_encoding(logpath)
    envname = get_envname(logpath)
    return color_and_style(encoding, backend, envname)

def color_and_style(encoding, backend, envname):
    color = None
    style = None
    if backend is not None:
        color = backend_colors[backend]
    if encoding is not None:
        style = encoding_styles[encoding]
    # if no backend, look if at least the envname is known
    if color is None:
        if envname in envname_colors:
            color = envname_colors[envname]
    return color, style

def get_backend_and_encoding(logpath):
    string = logpath.split("PPO_")[-1]
    backend = None
    encoding = None
    # find backend in string
    for k in backend_colors:
        if string.startswith(k):
            backend = k
            string = string.split(k+"_")[1]
            break
    # find encoding in remainder string
    for k in encoding_styles:
        if string.startswith(k):
            encoding = k
            break
    return backend, encoding

def get_envname(logpath):
    return os.path.basename(logpath).split("_")[0]

def get_label(envname, backend, encoding):
    label = None
    if envname in envname_labels:
        label = envname_labels[envname]
    else:
        label = "{}, {}".format(backend_labels[backend], encoding_labels[encoding])
    return label

def get_date(logpath):
    string = "_".join(os.path.basename(logpath).split("_")[1:])
    string = string.split("_PPO")[0]
    return string


if __name__ == "__main__":
    args, _ = parse_plotting_args()

    if args.scenario is not None:
        SCENARIO_AXES = {args.scenario : (0, 0)}  # show only 1 scenario
        ERROR_IF_MISSING = False
    if args.scenario == "navreptrain":
        SCENARIO_AXES = {args.scenario : (0, 0), "navrepval" : (1, 0)}
    if args.x_axis is None:
        args.x_axis = "train_steps"

    while True:
        logdir = args.logdir
        if logdir is None:
            logdir = "~/navrep"
        logdirs = [os.path.expanduser(logdir),]
        plot_per_scenario_training_progress(logdirs, SCENARIO_AXES, ERROR_IF_MISSING,
                                            x_axis=args.x_axis)
#         plot_per_scenario_training_progress(logdirs, SCENARIO_AXES, ERROR_IF_MISSING)
        plt.pause(60)
