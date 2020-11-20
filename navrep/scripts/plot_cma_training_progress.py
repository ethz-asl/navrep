from __future__ import print_function
import numpy as np
import json
import os
import matplotlib.pyplot as plt


env_name = "crowdnav"
optimizer = "cma"
num_rollouts = 16  # number of rollouts that are averaged over an episode
popsize = 32
required_score = 100.0


basedir = os.path.expanduser("~/navrep/models/cma")
logdirs = [x for x in os.listdir(basedir) if os.path.isdir(os.path.join(basedir, x))]
logdirs = sorted(logdirs)

all_data = []
all_best_data = []
titles = []
for logdir in logdirs:
    file_base = (
        env_name + "." + optimizer + "." + str(num_rollouts) + "." + str(popsize)
    )
    filename = os.path.join(basedir, logdir, file_base + ".hist.json")
    with open(filename, "r") as f:
        raw_data = json.load(f)
    data = np.array(raw_data)
    all_data.append(data)
    titles.append(filename)

    try:
        file_base = (
            env_name + "." + optimizer + "." + str(num_rollouts) + "." + str(popsize)
        )
        filename = os.path.join(basedir, logdir, file_base + ".hist_best.json")
        with open(filename, "r") as f:
            raw_data = json.load(f)
        raw_best_data = np.array(raw_data)
        best_data = []
        for bdata in raw_best_data:
            best_data.append(
                [
                    float(bdata[0]),
                    float(bdata[1]),
                    float(bdata[5]),
                    float(bdata[9]),
                    required_score,
                ]
            )
        best_data = np.array(best_data)
    except:
        best_data = np.zeros((10, 5)) * np.nan
    all_best_data.append(best_data)

    #     print(data)

    fig = plt.figure(figsize=(16, 10), dpi=80, facecolor="w", edgecolor="k")
    (line_mean,) = plt.plot(data[:, 1] / (60 * 24 * 60), data[:, 2])
    (line_min,) = plt.plot(data[:, 1] / (60 * 24 * 60), data[:, 3])
    (line_max,) = plt.plot(data[:, 1] / (60 * 24 * 60), data[:, 4])
    (line_best,) = plt.plot(best_data[:, 1] / (60 * 24 * 60), best_data[:, 2])
    (line_req,) = plt.plot(best_data[:, 1] / (60 * 24 * 60), best_data[:, 4])
    plt.axhline(
        color="k", zorder=-100,
    )
    plt.axhline(
        required_score, color="purple", zorder=-100,
    )
    plt.legend(
        [line_mean, line_min, line_max, line_req, line_best],
        ["mean", "min", "max", "requirement", "best avg score"],
    )
    plt.xlabel("wall-clock time (days)")
    plt.xticks(np.arange(0, 5, 1))
    plt.ylabel("cumulative reward")
    #     plt.yticks(np.arange(-100, 1000, 50))
    plt.title(filename)
    plt.show()

    fig = plt.figure(figsize=(16, 10), dpi=80, facecolor="w", edgecolor="k")
    (line_mean,) = plt.plot(data[:, 0], data[:, 2])
    (line_min,) = plt.plot(data[:, 0], data[:, 3])
    (line_max,) = plt.plot(data[:, 0], data[:, 4])
    (line_best,) = plt.plot(best_data[:, 0], best_data[:, 2])
    (line_req,) = plt.plot(best_data[:, 0], best_data[:, 4])
    plt.axhline(
        color="k", zorder=-100,
    )
    plt.axhline(
        required_score, color="purple", zorder=-100,
    )
    plt.legend(
        [line_mean, line_min, line_max, line_req, line_best],
        ["mean", "min", "max", "requirement", "best avg score"],
    )
    plt.xlabel("generation")
    #     plt.xticks(np.arange(0, 2000, 200))
    plt.ylabel("cumulative reward")
    #     plt.yticks(np.arange(-10, 1000, 10))
    plt.title(filename)
    plt.show()

fig = plt.figure(figsize=(16, 10), dpi=80, facecolor="w", edgecolor="k")
lines = []
legends = []
for data, best_data in zip(all_data, all_best_data):
    #     line, = plt.plot(data[:, 1]/(60*24*60), data[:, 2])
    #     line, = plt.plot(data[:, 1]/(60*24*60), data[:, 3])
    (line,) = plt.plot(data[:, 1] / (60 * 24 * 60), data[:, 4], alpha=0.6)
    lines.append(line)
    legends.append(filename)
    (line,) = plt.plot(
        best_data[:, 1] / (60 * 24 * 60),
        best_data[:, 2],
        alpha=0.6,
        color=line.get_color(),
    )
plt.axhline(
    color="k", zorder=-100,
)
plt.axhline(
    required_score, color="purple", zorder=-100,
)
plt.legend(lines, legends)
plt.xlabel("wall-clock time (days)")
plt.xticks(np.arange(0, 5, 1))
plt.ylabel("cumulative reward")
#     plt.yticks(np.arange(-100, 1000, 50))
# plt.title(filename)
plt.show()
