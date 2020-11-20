import numpy as np
from matplotlib import pyplot as plt

O_TRAJ_PROB = 0.01  # 1 means all trajectories are shown. lowering this number randomly removes trajs
S = 30  # markersize in points


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    # from stackoverflow how-to-extract-a-subset-of-a-colormap-as-a-new-colormap-in-matplotlib
    from matplotlib import colors
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def blue(u=np.random.rand()):
    cmap = truncate_colormap(plt.cm.bwr, 0., .4)
    u = np.clip(u, 0., 1.)
    return cmap(u)

def orange(u=np.random.rand()):
    cmap = truncate_colormap(plt.cm.Wistia, .3, .7)
    u = np.clip(u, 0., 1.)
    return cmap(u)

def plot_ianenv_statistics(S, ianenv, focus_scenario=None):
    plt.ion()
    if focus_scenario is None:
        scenarios = sorted(list(set(S["scenario"].values)))
    else:
        scenarios = [focus_scenario]
    if len(scenarios) > 6:
        fig, axes = plt.subplots(5, 6)
    else:
        fig, axes = plt.subplots(len(scenarios), 1)
    axes = np.array(axes).reshape((-1))
    for i, scenario in enumerate(scenarios):
        is_scenario = S["scenario"].values == scenario
        scenario_S = S[is_scenario]
        # print results
        goal_reached = scenario_S["goal_reached"].values
        goal_reached_perc = np.sum(goal_reached) * 100. / len(goal_reached)
        timed_out = scenario_S["timed_out"].values
        timed_out_perc = np.sum(timed_out) * 100. / len(timed_out)
        collisioned_out = scenario_S["collisioned_out"].values
        collisioned_out_perc = np.sum(collisioned_out) * 100. / len(collisioned_out)
        print("{} -----------------------------------".format(scenario))
        print("Goal reached: {}%".format(goal_reached_perc))
        print("Timed out: {}%".format(timed_out_perc))
        print("Collisioned out: {}%".format(collisioned_out_perc))

        # plot result
        trajectories = scenario_S["trajectory"]
        o_trajectories = scenario_S["other_trajectories"]
        goals = scenario_S["goal"]

        ax = axes[i]

        # TODO make this a map2d utility function
        # TODO get scenario map
        ianenv.reset(set_scenario=scenario)
        contours = ianenv.iarlenv.rlenv.virtual_peppers[0].map2d.as_closed_obst_vertices()
        for c in contours:
            cplus = np.concatenate((c, c[:1, :]), axis=0)
            ax.plot(cplus[:,0], cplus[:,1], color='k')
        # ^^^^

        for t, g, s, ot in zip(trajectories, goals, goal_reached, o_trajectories):
            line_color = blue(len(t)/1000.) if s else orange(len(t)/1000.)
            zorder = 2 if s else 1
            ax.plot(t[:,0], t[:,1], color=line_color, zorder=zorder)
            for o in ot:
                if len(o) > 0:
                    if np.random.rand() < O_TRAJ_PROB:
                        for xy in o:
                            ax.add_artist(plt.Circle((xy), 0.3,
                                                     edgecolor="dimgray", facecolor="none",
                                                     alpha=0.1, zorder=0))
            ax.add_artist(plt.Circle((t[0,0], t[0,1]), 0.3, color="green", zorder=2))
            ax.add_artist(plt.Circle((g[0], g[1]), 0.3, color="red", zorder=2))
        if focus_scenario is None:
            ax.set_title(scenario)
        else:
            ax.set_title("{} - {} - {}%".format(scenario, log, goal_reached_perc))
        ax.axis("equal")
        ax.set_adjustable('box')
        if "rl" in scenario:
            xlim = (np.min([np.min(t[:,0]) for t in trajectories]) - 3,
                    np.max([np.max(t[:,0]) for t in trajectories]) + 3)
            ylim = (np.min([np.min(t[:,1]) for t in trajectories]) - 3,
                    np.max([np.max(t[:,1]) for t in trajectories]) + 3)
            ax.set(xlim=xlim, ylim=ylim)

    goal_reached = S["goal_reached"].values
    goal_reached_perc = np.sum(goal_reached) * 100. / len(goal_reached)
    timed_out = S["timed_out"].values
    timed_out_perc = np.sum(timed_out) * 100. / len(timed_out)
    collisioned_out = S["collisioned_out"].values
    collisioned_out_perc = np.sum(collisioned_out) * 100. / len(collisioned_out)
    print("All scenarios -----------------------------------")
    print("Goal reached: {}%".format(goal_reached_perc))
    print("Timed out: {}%".format(timed_out_perc))
    print("Collisioned out: {}%".format(collisioned_out_perc))
    plt.pause(1.)


if __name__ == "__main__":
    import os
    import pandas as pd

    from navrep.tools.commonargs import parse_plotting_args
    from navrep.envs.ianenv import IANEnv

    args, _ = parse_plotting_args()

    LOGDIR = args.logdir
    if LOGDIR is None:
        LOGDIR = "~/navrep"
    LOGDIR = os.path.join(LOGDIR, "eval/crosstest")
    LOGDIR = os.path.expanduser(LOGDIR)
    logs = sorted(os.listdir(LOGDIR))
    for log in logs:
        if not log.endswith(".pckl"):
            continue
        print()
        print(log)
        path = os.path.join(LOGDIR, log)
        S = pd.read_pickle(path)

        env = IANEnv(silent=True)

        plot_ianenv_statistics(S, env, focus_scenario=args.scenario)
