import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib

from navrep.scripts.plot_gym_training_progress import get_backend_and_encoding, get_envname, get_date, \
    get_color_and_style, color_and_style, get_label
from navrep.tools.commonargs import parse_plotting_args

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 17

ALL_RUNS = False

backend_ranks = {
    "VAE1D_LSTM": 0,
    "VAE_LSTM": 3,
    "VAE1DLSTM": 1,
    "VAELSTM": 4,
    "GPT1D": 2,
    "GPT": 5,
    None: 0,
}
envname_ranks = {
    "e2e1dnavreptrain": 0,
    "e2enavreptrain": 1,
    "lucianavreptrain": 2,
    "navreptrainencodedenv": 3,
}
encoding_ranks = {
    "V_ONLY": 0,
    "M_ONLY": 1,
    "VM": 2,
    None: 0,
}
def sort_by_env_backend_encoding_date(logs):
    sortkeys = {}
    for log in logs:
        backend, encoding = get_backend_and_encoding(log)
        envname = get_envname(log)
        date = get_date(log)
        try:
            sortkeys[log] = (envname_ranks[envname], backend_ranks[backend], encoding_ranks[encoding], date)
        except KeyError as e:
            print(e)
            sortkeys[log] = (0, 0, 0, '')

    sorted_logs = sorted(logs, key=lambda x:sortkeys[x])
    return sorted_logs


def style_to_hatch(style):
    hatch = None
    if style == "dotted":
        hatch = "\\"
    if style == "dashed":
        hatch = "//"
    if style == "solid":
        hatch = None
    return hatch

def calculate_bar_offsets(bar_width, n_bars):
    offsets = np.linspace(bar_width / 2. - 0.8 / 2., 0.8 / 2 - bar_width / 2, n_bars)
    return offsets


if __name__ == "__main__":
    args, _ = parse_plotting_args()

    LOGDIR = args.logdir
    if LOGDIR is None:
        LOGDIR = "~/navrep"
    LOGDIR = os.path.join(LOGDIR, "eval/crosstest")
    LOGDIR = os.path.expanduser(LOGDIR)
#     logs = [log for log in os.listdir(LOGDIR) if log.endswith(".pckl")]
    logs = [log for log in os.listdir(LOGDIR) if log.endswith(".csv")]
    logs = sort_by_env_backend_encoding_date(logs)
    Ss = []
    log_scenarios = []
    log_success_rates = []
    for log in logs:
        path = os.path.join(LOGDIR, log)
        S = pd.read_csv(path)
#         S = pd.read_pickle(path)

        print()
        print()
        print(" ----------------------------------")
        print(" | {:^30.30} |".format(log))
        print(" ----------------------------------")
        print()

        all_success_rate = np.mean(S["goal_reached"])
        all_avg_reward = np.mean(S["reward"])
        print("ALL: {:.1f}% - R={:.1f}".format(100.*all_success_rate, all_avg_reward))
        print("-----------")

        scenarios = np.array(sorted(list(set(S["scenario"]))))
        success_rates = []
        for scenario in scenarios:
            scenario_specific = S[S["scenario"] == scenario]
            success_rate = np.mean(scenario_specific["goal_reached"])
            avg_reward = np.mean(scenario_specific["reward"])
            n = len(scenario_specific["reward"])
            print("{} ({}): {:.1f}%".format(scenario, n, 100.*success_rate))
            success_rates.append(success_rate)

        Ss.append(S)
        log_scenarios.append(scenarios)
        log_success_rates.append(success_rates)

    all_scenarios = sorted(list(set([s for l_ in log_scenarios for s in l_])))
    scenario_idx = {s: i for i, s in enumerate(all_scenarios)}

    if ALL_RUNS:
        # Plot per scenario

        fig, ax = plt.subplots()
        width = 0.8 / len(log_scenarios)
        offsets = calculate_bar_offsets(width, len(log_scenarios))

        for legend, x_labels, y_values, o in zip(logs, log_scenarios, log_success_rates, offsets):
            x = np.array([scenario_idx[s] for s in x_labels])
            color, style = get_color_and_style(legend)
            hatch = style_to_hatch(style)
            plt.bar(x + o, y_values, width, color=color, edgecolor="white", hatch=hatch, label=legend)

        xticks = range(len(all_scenarios))
        plt.xticks(xticks, all_scenarios, rotation='vertical')
        plt.legend(bbox_to_anchor=(1.05, 1.))

    if ALL_RUNS:
        # Group per map
        all_groups = ["asl", "unity_scene_map", "asl_office_j"]
        group_success_rates = []
        for S in Ss:
            success_rates = []
            for group in all_groups:
                if group == "asl":
                    scenario_specific = S[~S["scenario"].str.contains("asl_office_j")
                                          & ~S["scenario"].str.contains("unity_scene_map")]
                else:
                    scenario_specific = S[S["scenario"].str.contains(group)]
                success_rate = np.mean(scenario_specific["goal_reached"])
                success_rates.append(success_rate)
            group_success_rates.append(success_rates)

        fig, ax = plt.subplots()
        for legend, y_values, o in zip(logs, group_success_rates, offsets):
            x_labels = all_groups
            x = np.arange(len(x_labels))
            color, style = get_color_and_style(legend)
            hatch = style_to_hatch(style)
            plt.bar(x + o, y_values, width, color=color, edgecolor="white", hatch=hatch, label=legend)
        xticks = range(len(all_groups))
        plt.xticks(xticks, all_groups, rotation='vertical')
        plt.legend(bbox_to_anchor=(1.05, 1.))

    if ALL_RUNS:
        # Total
        total_success_rates = []
        for S in Ss:
            success_rate = np.mean(S["goal_reached"])
            total_success_rates.append([success_rate])

        fig, ax = plt.subplots()
        for legend, y_values, o in zip(logs, total_success_rates, offsets):
            x_labels = ["all scenarios"]
            x = np.arange(len(x_labels))
            color, style = get_color_and_style(legend)
            hatch = style_to_hatch(style)
            plt.bar(x + o, y_values, width, color=color, edgecolor='white', hatch=hatch, label=legend)
        xticks = range(len(x_labels))
        plt.xticks(xticks, x_labels, rotation='vertical')
        plt.legend(bbox_to_anchor=(1.05, 1.))

    # ----------------------------------------------------------
    # Paper

    # calculate specie success rates
    # group per encoding
    species_success_rates = {}
    map_groups = ["all", "asl", "unity_scene_map", "asl_office_j"]
    scenario_groups = all_scenarios
    all_groups = map_groups + scenario_groups
    for log, S in zip(logs, Ss):
        backend, encoding = get_backend_and_encoding(log)
        envname = get_envname(log)
        specie_key = (envname, backend, encoding)
        if specie_key not in species_success_rates:
            species_success_rates[specie_key] = {group: [] for group in all_groups}

        # get per-map success rate
        for group in map_groups:
            if group == "asl":
                group_specific = S[~S["scenario"].str.contains("asl_office_j")
                                   & ~S["scenario"].str.contains("unity_scene_map")]
            elif group == "all":
                group_specific = S
            else:
                group_specific = S[S["scenario"].str.contains(group)]
            success_rate = np.mean(group_specific["goal_reached"])

            species_success_rates[specie_key][group].append(success_rate)

        # get per-scenario success rate
        for group in scenario_groups:
            group_specific = S[S["scenario"] == group]
            success_rate = np.mean(group_specific["goal_reached"])

            species_success_rates[specie_key][group].append(success_rate)

    # table
    table_groups = [
        "irosasl2",
        "irosasl6",
        "irosasl3",
        "irosasl1",
        "irosasl4",
        "irosasl5",
        "rlasl1",
        "rlasl2",
        "rlasl3",
        "irosunity_scene_map2",
        "irosunity_scene_map6",
        "irosunity_scene_map3",
        "irosunity_scene_map1",
        "irosunity_scene_map4",
        "irosunity_scene_map5",
        "rlunity_scene_map1",
        "rlunity_scene_map2",
        "rlunity_scene_map3",
        "irosasl_office_j2",
        "irosasl_office_j6",
        "irosasl_office_j3",
        "irosasl_office_j1",
        "irosasl_office_j4",
        "irosasl_office_j5",
        "rlasl_office_j1",
        "rlasl_office_j2",
        "rlasl_office_j3",
    ]
    group_labels = {
        "irosasl2": "1",
        "irosasl6": "2",
        "irosasl3": "3",
        "irosasl1": "4",
        "irosasl4": "5",
        "irosasl5": "6",
        "rlasl1": "7",
        "rlasl2": "8",
        "rlasl3": "9",
        "irosunity_scene_map2": "1",
        "irosunity_scene_map6": "2",
        "irosunity_scene_map3": "3",
        "irosunity_scene_map1": "4",
        "irosunity_scene_map4": "5",
        "irosunity_scene_map5": "6",
        "rlunity_scene_map1": "7",
        "rlunity_scene_map2": "8",
        "rlunity_scene_map3": "9",
        "irosasl_office_j2": "1",
        "irosasl_office_j6": "2",
        "irosasl_office_j3": "3",
        "irosasl_office_j1": "4",
        "irosasl_office_j4": "5",
        "irosasl_office_j5": "6",
        "rlasl_office_j1": "7",
        "rlasl_office_j2": "8",
        "rlasl_office_j3": "9",
    }
    group_best = {group: -np.inf for group in table_groups}
    for specie_key in species_success_rates:
        specie_success_rates = species_success_rates[specie_key]
        for group in table_groups:
            success = np.max(specie_success_rates[group]) * 100.
            if success > group_best[group]:
                group_best[group] = success
    print()
    print("Success rates LaTEX table:")
    print("--------------------------")
    c_s = " ".join(["c"] * (len(table_groups) + 1))
    print('\\begin{tabular}{ ' + "{}".format(c_s) + "}")
    print('\\toprule')
    print('& \\multicolumn{9}{c}{simple} & \\multicolumn{9}{|c|}{complex} & \\multicolumn{9}{c}{realistic} \\' '\\')  # noqa
    for group in table_groups:
        group_label = group_labels[group]
        print("& {}    ".format(group_label), end="")
    print('\\'+'\\')
    print('\\midrule')
    for specie_key in species_success_rates:
        envname, backend, encoding = specie_key
        specie_label = get_label(envname, backend, encoding)
        specie_success_rates = species_success_rates[specie_key]

        print("{:<10} ".format(specie_label), end="")
        for group in table_groups:
            n_trials = len(specie_success_rates[group])
            planner_scene_avgsr = "--"
            if n_trials != 0:
                success = np.max(specie_success_rates[group]) * 100.
                planner_scene_avgsr = "{:.1f}".format(success)
                if success == group_best[group] and success != 0:
                    planner_scene_avgsr = '\\textbf{' + planner_scene_avgsr + '}'
            print("& {:>10} ".format(planner_scene_avgsr), end="")
        print('\\'+'\\')
    print('\\bottomrule')
    print('\\end{tabular}')
    print()

    # box plot
    width = 0.8 / len(species_success_rates)
    offsets = calculate_bar_offsets(width, len(species_success_rates))

    fig, ax = plt.subplots()
    for key, o in zip(species_success_rates, offsets):
        envname, backend, encoding = key
        label = get_label(envname, backend, encoding)
        specie_success_rates = species_success_rates[key]
        ymin = np.array([np.min(specie_success_rates[group]) for group in map_groups])
        ymax = np.array([np.max(specie_success_rates[group]) for group in map_groups])
        ymean = np.array([np.mean(specie_success_rates[group]) for group in map_groups])
        yn = np.array([len(specie_success_rates[group]) for group in map_groups])
        x_labels = map_groups
        x = np.arange(len(x_labels))
        color, style = color_and_style(encoding, backend, envname)
        hatch = style_to_hatch(style)
        BOX_PLOT = False
        if BOX_PLOT:
            ymin = ymin - 0.005
            ymax = ymax + 0.005
            plt.bar(x + o, ymax-ymin, width, bottom=ymin, # box plot
                    color=color, edgecolor="white", hatch=hatch, label=label)
            for xi, ymeani in zip(x, ymean):
                plt.plot([xi + o - width/2., xi + o + width/2.], [ymeani, ymeani], color='dimgray')
        else:
            plt.bar(x + o, ymean, width,
                    color=color, edgecolor="white", hatch=hatch, label=label)
            # error bars
            for xi, ymini, ymaxi, yni in zip(x, ymin, ymax, yn):
                if yni > 1:
                    plt.plot([xi + o, xi + o], [ymini, ymaxi], color='dimgray')
    xticks = np.arange(len(map_groups))
    vlines = xticks[:-1] + 0.5
    for vline in vlines:
        plt.axvline(vline, color='dimgray', linewidth=1)
    plt.xticks(xticks, map_groups, rotation='vertical')
    plt.legend(bbox_to_anchor=(1.05, 1.))

    plt.show()
