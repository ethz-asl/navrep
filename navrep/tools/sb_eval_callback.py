import numpy as np
import time
from pandas import DataFrame
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.common.vec_env import VecEnv

from navrep.scripts.test_navrep import run_test_episodes, NavRepCPolicy


class NavrepLogCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param logpath: (string) where to save the training log
    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, logpath=None, savepath=None, eval_freq=10000, verbose=0):
        super(NavrepLogCallback, self).__init__(verbose)
        # self.model = None  # type: BaseRLModel
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        self.logpath = logpath
        self.savepath = savepath
        self.eval_freq = eval_freq
        self.last_len_statistics = 0
        self.best_avg_reward = [-np.inf]

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        :return: (bool) If the callback returns False, training is aborted early.
        """
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # get episode_statistics
            env = self.training_env
            if isinstance(self.training_env, VecEnv):
                S = DataFrame()
                for env in self.training_env.envs:
                    S = S.append(env.episode_statistics, ignore_index=True)
                S["total_steps"] *= len(self.training_env.envs)
                S = S.sort_values(by="total_steps", ignore_index=True)
            else:
                S = env.episode_statistics

            new_S = S[self.last_len_statistics:]
            new_avg_reward = np.mean(new_S["reward"].values)

            save_log(S, self.logpath, self.verbose)
            print_statistics(new_S, 0, 0, self.n_calls, self.num_timesteps, self.verbose)
            save_model_if_improved(new_avg_reward, self.best_avg_reward, self.model, self.savepath)

            self.last_len_statistics = len(S)
        return True


class NavrepEvalCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param eval_env: (gym.Env) environment with which to evaluate the model (at eval_freq)
    :param test_env_fn: (function) function which returns an environment which is used to evaluate
                        the model after every tenth evaluation.
    :param n_eval_episodes: (int) how many episodes to run the evaluation env for
    :param logpath: (string) where to save the training log
    :param savepath: (string) where to save the model
    :param eval_freq: (int) how often to run the evaluation
    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    :param render: (bool) human rendering in the test env
    """
    def __init__(self, eval_env, test_env_fn=None,
                 n_eval_episodes=20, logpath=None, savepath=None, eval_freq=10000, verbose=0,
                 render=False):
        super(NavrepEvalCallback, self).__init__(verbose)
        # self.model = None  # type: BaseRLModel
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        self.logpath = logpath
        self.savepath = savepath
        self.eval_freq = eval_freq
        self.last_len_statistics = 0
        self.best_avg_reward = [-np.inf]
        self.eval_env = eval_env
        self.test_env_fn = test_env_fn
        self.last_eval_time = time.time()
        self.render = render

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        :return: (bool) If the callback returns False, training is aborted early.
        """
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0 or self.n_calls == 1:
            # get episode_statistics
            tic = time.time()
            S = run_k_episodes(self.model, self.eval_env, 20)
            toc = time.time()
            eval_duration = toc - tic

            new_avg_reward = -np.inf
            # every 10 evaluations, do a more thorough test
            if self.n_calls % (self.eval_freq * 10) == 0 or self.n_calls == 1:
                if self.test_env_fn is not None:
                    policy = NavRepCPolicy(model=self.model)
                    test_env = self.test_env_fn()
                    success_rate, avg_nav_time = run_test_episodes(test_env, policy,
                                                                   num_episodes=100, render=self.render)
                    # we add a first point to the history log, for nice plotting
                    S.loc[len(S)] = [
                        self.eval_env.total_steps,
                        'navrepval',
                        0,
                        avg_nav_time / test_env._get_dt(),
                        success_rate,
                        success_rate * 100.,
                        test_env.soadrl_sim.human_num,
                        test_env.soadrl_sim.num_walls,
                        time.time(),
                    ]
                    new_avg_reward = success_rate

            new_S = S[self.last_len_statistics:]
            if self.test_env_fn is None:
                new_avg_reward = np.mean(new_S["reward"].values)

            elapsed = time.time() - self.last_eval_time
            self.last_eval_time = time.time()

            save_log(S, self.logpath, self.verbose)
            print_statistics(new_S, elapsed, eval_duration, self.n_calls, self.num_timesteps, self.verbose)
            save_model_if_improved(new_avg_reward, self.best_avg_reward, self.model, self.savepath)

            self.last_len_statistics = len(S)
        return True

def save_log(S, logpath, verbose):
    # save log
    if logpath is not None:
        S.to_csv(logpath)
        if verbose > 1:
            print("log saved to {}.".format(logpath))

def print_statistics(S, elapsed, eval_elapsed, n_calls, n_train_steps, verbose):
    scenarios = sorted(list(set(S["scenario"].values)))
    rewards = S["reward"].values
    # print statistics
    if verbose > 0:
        print("Step {} (Env) {} (Train) - {} completed episodes - {:.0f}s (All) {:.0f}s (Eval)".format(
            n_calls, n_train_steps, len(S), elapsed, eval_elapsed))
        for scenario in scenarios:
            is_scenario = S["scenario"].values == scenario
            scenario_rewards = rewards[is_scenario]
            print("{}: {:.4f} ({})".format(
                scenario, np.mean(scenario_rewards), len(scenario_rewards)))

def save_model_if_improved(new_avg_reward, best_avg_reward, model, savepath):
    if new_avg_reward > best_avg_reward[0]:
        best_avg_reward[0] = new_avg_reward
        if savepath is not None:
            try:
                model.save(savepath)
                print("model saved to {} (avg reward: {}).".format(
                    savepath, new_avg_reward))
            except:  # noqa
                print("Could not save")

def run_k_episodes(model, env, k):
    for i in range(k):
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = env.step(action)
    return env.episode_statistics
