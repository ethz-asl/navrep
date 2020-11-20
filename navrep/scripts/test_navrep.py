import os
import numpy as np
from tqdm import tqdm
from stable_baselines import PPO2
from navrep.envs.navreptrainencodedenv import NavRepTrainEncodedEnv

from crowd_sim.envs.utils.info import Timeout, ReachGoal, Danger, Collision, CollisionOtherAgent
from navrep.tools.commonargs import parse_common_args

class NavRepCPolicy(object):
    """ wrapper for gym policies """
    def __init__(self, model=None):
        if model is not None:
            self.model = model
        else:
            self.model_path = os.path.expanduser(
                "~/navrep/models/gym/navreptrainencodedenv_latest_PPO_ckpt")
            self.model = PPO2.load(self.model_path)
            print("Model '{}' loaded".format(self.model_path))

    def act(self, obs):
        action, _states = self.model.predict(obs, deterministic=True)
        return action


def run_test_episodes(env, policy, render=False, print_failure=True, num_episodes=500):
    success_times = []
    collision_times = []
    collision_other_agent_times = []
    timeout_times = []
    success = 0
    collision = 0
    collision_other_agent = 0
    timeout = 0
    too_close = 0
    min_dist = []
    cumulative_rewards = []
    collision_cases = []
    collision_other_agent_cases = []
    timeout_cases = []
    progress_bar = tqdm(range(num_episodes), total=num_episodes)
    for i in progress_bar:
        progress_bar.set_description("Case {}".format(i))
        ob = env.reset()
        done = False
        env_time = 0
        while not done:
            action = policy.act(ob)
            ob, _, done, info = env.step(action)
            event = info['event']
            if render:
                env.render('human')  # robocentric=True, save_to_file=True)
            env_time += env._get_dt()
            if isinstance(event, Danger):
                too_close += 1
                min_dist.append(event.min_dist)

        if isinstance(event, ReachGoal):
            success += 1
            success_times.append(env_time)
        elif isinstance(event, Collision):
            collision += 1
            collision_cases.append(i)
            collision_times.append(env_time)
        elif isinstance(event, CollisionOtherAgent):
            collision_other_agent += 1
            collision_other_agent_cases.append(i)
            collision_other_agent_times.append(env_time)
        elif isinstance(event, Timeout):
            timeout += 1
            timeout_cases.append(i)
            timeout_times.append(env_time)
        else:
            raise ValueError('Invalid end signal from environment')

    success_rate = success / float(num_episodes)
    collision_rate = collision / float(num_episodes)
    collision_other_agent_rate = collision_other_agent / \
        float(num_episodes)
    assert success + collision + timeout + collision_other_agent == num_episodes
    avg_nav_time = sum(success_times) / float(len(success_times)
                                              ) if success_times else np.nan

    print(
        """has success rate: {:.2f}, collision rate: {:.2f},
        collision from other agents rate: {:.2f}, nav time: {:.2f}, total reward: {:.4f}""".format(
            success_rate,
            collision_rate,
            collision_other_agent_rate,
            avg_nav_time,
            np.mean(cumulative_rewards)
        )
    )
    total_time = sum(success_times + collision_times + collision_other_agent_times + timeout_times)
    print(
        'Frequency of being in danger: %.2f and average min separate distance in danger: %.2f',
        too_close / float(total_time),
        np.mean(min_dist))

    if print_failure:
        print('Collision cases: ' + ' '.join([str(x) for x in collision_cases]))
        print('Collision from other agent cases: ' + ' '.join([str(x) for x in collision_other_agent_cases]))
        print('Timeout cases: ' + ' '.join([str(x) for x in timeout_cases]))

    return success_rate, avg_nav_time


if __name__ == '__main__':
    args, _ = parse_common_args()

    if args.environment is None or args.environment == "navreptrain":
        env = NavRepTrainEncodedEnv(args.backend, args.encoding, silent=True, scenario='test')
        policy = NavRepCPolicy()
    else:
        raise NotImplementedError

    run_test_episodes(env, policy, render=args.render)
