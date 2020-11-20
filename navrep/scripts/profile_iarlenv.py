#!/usr/bin/env python
import gc
import numpy as np
import os
from pepper_2d_iarlenv import parse_iaenv_args, IARLEnv, check_iaenv_args
from pyniel.python_tools.timetools import WalltimeRate
from timeit import default_timer as timer

from navrep.envs.scenario_list import set_scenario, scenarios


if __name__ == "__main__":
    for scenario_name in scenarios:
        ## setting up the sim ---------------
        tic = timer()
        # args
        args = parse_iaenv_args()
        args.unmerged_scans = False
        args.continuous = True
        args.naive_plan = True
        args.no_ros = True
        set_scenario(args, scenario_name=scenario_name)
        check_iaenv_args(args)
        # env
        iarlenv = IARLEnv(args, silent=True)
        #         gc.collect()
        iarlenv.reset()
        action = np.array([0.0, 0.0, 0.0])
        toc = timer()
        setup_time = toc - tic
        tic = timer()
        N_STEPS = 1000
        for i in range(N_STEPS):
            obs, rew, done, _ = iarlenv.step(action, ONLY_FOR_AGENT_0=True)
        toc = timer()
        episode_time = toc - tic
        print(
            "Scenario {} - {} steps - setup: {:.1f}s - episode {:.1f}s".format(
                scenario_name, N_STEPS, setup_time, episode_time
            )
        )
