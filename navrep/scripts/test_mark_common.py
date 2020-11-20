import os
import numpy as np
import pandas as pd
import time
from stable_baselines import PPO2
from matplotlib import pyplot as plt
from pyniel.python_tools.path_tools import make_dir_if_not_exists

from navrep.tools.commonargs import parse_multiproc_args
from navrep.envs.markenv import FIRST_TEST_MAPS
from navrep.envs.markencodedenv import MarkOneEncodedEnv


class MarkOneVMCPolicy(object):
    """
    Compatible with raw lidar envs (IANEnv, MarkEnv),
    takes a (1080 lidar, 5 state) obs as input,
    converts it for gym C,
    outputs (3,) action"""
    def __init__(self):
        self.model_path = os.path.expanduser(
            "~/navrep/models/gym/markoneencodedenv_latest_PPO_ckpt")
        self.encoder_env = MarkOneEncodedEnv(silent=True)
        self.prev_action = np.array([0.,0.,0.])
        self.model = PPO2.load(self.model_path)
        print("Model '{}' loaded".format(self.model_path))

    def act(self, obs):
        vm_obs = self.encoder_env.encoder._encode_obs(obs, self.prev_action)
        action, _states = self.model.predict(vm_obs, deterministic=True)
        action = np.array([action[0], action[1], 0.])  # no rotation
        self.prev_action = action * 1.
        return action

class MarkOneCPolicy(object):
    """
    Compatible with encoded envs (EncodedEnv, MarkOneEncodedEnv, etc...)
    takes a (546) obs as input,
    converts it for gym C,
    outputs (2,) action"""
    def __init__(self):
        self.model_path = os.path.expanduser(
            "~/navrep/models/gym/markoneencodedenv_latest_PPO_ckpt")
        self.model = PPO2.load(self.model_path)
        print("Model '{}' loaded".format(self.model_path))

    def act(self, obs):
        action, _states = self.model.predict(obs, deterministic=True)
        return action

class MarkTwoCPolicy(object):
    """Compatible with encoded envs (EncodedEnv, MarkOneEncodedEnv, etc...)
    takes a (546) obs as input,
    converts it for gym C,
    outputs (2,) action"""
    def __init__(self):
        self.model_path = os.path.expanduser(
            "~/navrep/models/gym/marktwoencodedenv_latest_PPO_ckpt")
        self.model = PPO2.load(self.model_path)
        print("Model '{}' loaded".format(self.model_path))

    def act(self, obs):
        action, _states = self.model.predict(obs, deterministic=True)
        return action

def markeval(policy, env, n_episodes=None,
             subset_index=0, n_subsets=1,
             render=False,
             ):
    if n_episodes is None:
        n_episodes = 100

    episode_ids = np.arange(n_episodes)
    if n_subsets > 1:  # when multiprocessing
        episode_ids = np.array_split(episode_ids, n_subsets)[subset_index]

    markeval_dir = "~/navrep/eval/markeval"
    stats_dir = os.path.join(os.path.expanduser(markeval_dir), os.path.basename(policy.model_path))
    make_dir_if_not_exists(stats_dir)
    stats_path = os.path.join(stats_dir, "episode_statistics_{}_to_{}.pckl".format(
        episode_ids[0], episode_ids[-1]))

    # Test for n episodes
    obs = env.reset()
    # run policy
    for i in episode_ids:
        while True:
            action = policy.act(obs)
            obs, rewards, done, info = env.step(action)
            if render:
                env.render()
            if done:
                env.reset()
                break
        print("Simulating episode {}/{}".format(i, n_episodes), end="\r")

    S = env.episode_statistics
    S.to_pickle(stats_path)

    if subset_index == 0:
        while True:
            all_S = load_markeval_statistics(stats_dir)
            if len(all_S) < n_episodes:
                time.sleep(1.)
            else:
                break

        plot_markeval_statistics(all_S, env)

    return env.episode_statistics

def clear_markeval_statistics(stats_dir):
    files = []
    for dirpath, dirnames, filenames in os.walk(stats_dir):
        for filename in [f for f in filenames if f.endswith(".pckl")]:
            files.append(os.path.join(dirpath, filename))
    files = sorted(files)
    if files:
        key = input("Directory {} already contains {} eval statistic files. Delete? [y/N]".format(
            stats_dir, len(files)))
        if key.lower() in ['y', 'yes']:
            for file in files:
                os.remove(file)
                print("{} removed.".format(file))
        else:
            raise ValueError("No way forward. (files already exist)")

def load_markeval_statistics(stats_dir):
    files = []
    for dirpath, dirnames, filenames in os.walk(stats_dir):
        for filename in [f for f in filenames if f.endswith(".pckl")]:
            files.append(os.path.join(dirpath, filename))
    files = sorted(files)
    S_parts = [pd.read_pickle(file) for file in files]
    all_S = pd.concat(S_parts, ignore_index=True)
    return all_S

def plot_markeval_statistics(S, env):
    plt.ion()
    scenarios = sorted(list(set(S["scenario"].values)))
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
        goals = scenario_S["goal"]

        ax = axes[i]

        # TODO make this a map2d utility function
        # TODO get scenario map
        env.reset(set_scenario=scenario)
        contours = env.rlenv.virtual_peppers[0].map2d.as_closed_obst_vertices()
        for c in contours:
            cplus = np.concatenate((c, c[:1, :]), axis=0)
            ax.plot(cplus[:,0], cplus[:,1], color='k')
        # ^^^^

        for t, g, s in zip(trajectories, goals, goal_reached):
            line_color = "blue" if s else "orange"
            ax.plot(t[:,0], t[:,1], color=line_color)
            ax.scatter(t[0,0], t[0,1], color="green")
            ax.scatter(g[0], g[1], color="red")
        ax.axis("equal")

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
    args, _ = parse_multiproc_args()

    # example usage
    env = MarkOneEncodedEnv(silent=True, maps=FIRST_TEST_MAPS)
    policy = MarkOneCPolicy()
    S = markeval(policy, env,
                 n_episodes=args.n,
                 subset_index=args.subproc_id, n_subsets=args.n_subprocs,
                 render=args.render)
