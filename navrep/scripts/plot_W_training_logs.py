import pandas as pd
from matplotlib import pyplot as plt
import os
import sys
import numpy as np

UPDATABLE = True

LOGDIR = os.path.expanduser("~/navrep/logs/W")
if len(sys.argv) == 2:
    LOGDIR = sys.argv[1]

plt.close('all')
while True:
    plt.figure("training log")
    plt.clf()
    logs = sorted(os.listdir(LOGDIR))
    legend = []
    lines = []
    for log in logs:
        if not log.endswith(".csv"):
            continue
        path = os.path.join(LOGDIR, log)
        legend.append(log)
        data = pd.read_csv(path)
        x = data["step"].values
        y = data["cost"].values
        line, = plt.plot(x, y, label=log)
        plt.axhline(np.min(y), alpha=0.3, linewidth=1, color=line.get_color())
        lines.append(line)
    plt.legend()   # quick-search : plot tcn training logs
    plt.ion()
    plt.pause(10.)
