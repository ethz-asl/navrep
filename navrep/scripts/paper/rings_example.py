import numpy as np
from matplotlib import pyplot as plt

from navrep.tools.rings import generate_rings


if __name__ == "__main__":
    ring_def = generate_rings(max_dist=10)
    # linear ramp
    data = np.load("/home/daniel/navrep/datasets/V/irl/rik_bananas_scans_robotstates_actions_rewards_dones.npz")
    scans = data["scans"][:1, :]
    scans[scans == 0] = np.inf
    rings = ring_def["lidar_to_rings"](scans.astype(np.float32))
    plt.ion()
    ring_def["visualize_rings"](rings[0, :, :, :], scan=scans[0])
