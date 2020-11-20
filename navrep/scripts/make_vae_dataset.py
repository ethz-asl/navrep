from pyniel.python_tools.path_tools import make_dir_if_not_exists
import numpy as np
import os
from crowd_sim.envs.policy.orca import ORCA
from crowd_sim.envs.utils.state import JointState

from navrep.tools.commonargs import parse_multiproc_args
from navrep.tools.data_extraction import folder_to_archive
from navrep.envs.toyenv import ToyEnv
from navrep.envs.markenv import MarkEnv, FIRST_TRAIN_MAPS, SECOND_TRAIN_MAPS
from navrep.envs.navreptrainenv import NavRepTrainEnv

class Suicide(object):
    def __init__(self):
        pass

class RandomMomentumPolicy(object):
    def __init__(self):
        self.speed = None

    def reset(self):
        self.speed = np.zeros((3,))

    def predict(self, obs, env):
        # create pseudo-random behavior
        acceleration = 0.1 * (np.random.random((3,)) - 0.5)
        friction = -self.speed * np.array([0.01, 0.01, 0.1])
        self.speed = np.clip(self.speed + acceleration + friction, -0.5, 0.5)
        return self.speed * 1.

class ORCAPolicy(object):
    def __init__(self, suicide_if_stuck=False):
        self.simulator = ORCA()
        self.suicide_if_stuck = suicide_if_stuck

    def reset(self):
        self.simulator.reset()

    def predict(self, obs, env):
        self.simulator.time_step = env._get_dt()
        other_agent_states = [
            agent.get_observable_state() for agent in env.soadrl_sim.humans + env.soadrl_sim.other_robots]
        action = self.simulator.predict(
            JointState(env.soadrl_sim.robot.get_full_state(), other_agent_states),
            env.soadrl_sim.obstacle_vertices,
            env.soadrl_sim.robot,
        )
        if self.suicide_if_stuck:
            if action.v < 0.1:
                return Suicide()
        vx = action.v * np.cos(action.r)
        vy = action.v * np.sin(action.r)
        return np.array([vx, vy, 0.1*(np.random.random()-0.5)])


def generate_vae_dataset(env, n_sequences,
                         episode_length=1000,
                         subset_index=0, n_subsets=1,
                         render=True,
                         policy=RandomMomentumPolicy(),
                         archive_dir=os.path.expanduser("~/navrep/datasets/V/ian")
                         ):
    """
    if n_subsets is None, the whole set of sequences is generated (n_sequences)
    if n_subsets is a number > 1, this function only generates a portion of the sequences
    """
    indices = np.arange(n_sequences)
    if n_subsets > 1:  # when multiprocessing
        indices = np.array_split(indices, n_subsets)[subset_index]
    for n in indices:
        scans = []
        robotstates = []
        actions = []
        rewards = []
        dones = []
        policy.reset()
        obs = env.reset()
        for i in range(episode_length):
            # step
            action = policy.predict(obs, env)
            if isinstance(action, Suicide):
                obs = env.reset()
                rew = 0
                action = np.array([0, 0, 0])
                done = True
            else:
                obs, rew, done, _ = env.step(action)
            scans.append(obs[0])
            robotstates.append(obs[1])
            actions.append(action)
            rewards.append(rew)
            dones.append(done)
            if render:
                env.render()
            if done:
                policy.reset()
                obs = env.reset()
            print("{} - {} {}".format(n, i, "done" if done else "     "), end="\r")
        dones[-1] = True

        scans = np.array(scans)
        robotstates = np.array(robotstates)
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones)
        data = dict(scans=scans, robotstates=robotstates, actions=actions, rewards=rewards, dones=dones)
        if archive_dir is not None:
            make_dir_if_not_exists(archive_dir)
            archive_path = os.path.join(
                archive_dir, "{:03}_scans_robotstates_actions_rewards_dones.npz".format(n)
            )
            np.savez_compressed(archive_path, **data)
            print(archive_path, "written.")
    env.close()
    return data


if __name__ == "__main__":
    args, _ = parse_multiproc_args()
    n_sequences = 100
    if args.n is not None:
        n_sequences = args.n

    if args.environment == "ian":
        # faster to use pre-existing ros bags than regenerate
        folder_to_archive("~/autoeval_bc")
    if args.environment == "toy":
        archive_dir = os.path.expanduser("~/navrep/datasets/V/toy")
        if args.dry_run:
            archive_dir = "/tmp/navrep/datasets/V/toy"
        env = ToyEnv(silent=True)
        generate_vae_dataset(
            env, n_sequences=n_sequences,
            subset_index=args.subproc_id, n_subsets=args.n_subprocs,
            render=args.render, archive_dir=archive_dir)
    if args.environment == "markone":
        archive_dir = os.path.expanduser("~/navrep/datasets/V/markone")
        if args.dry_run:
            archive_dir = "/tmp/navrep/datasets/V/markone"
        env = MarkEnv(silent=True, maps=FIRST_TRAIN_MAPS)
        generate_vae_dataset(
            env, n_sequences=n_sequences,
            subset_index=args.subproc_id, n_subsets=args.n_subprocs,
            render=args.render, archive_dir=archive_dir)
    if args.environment == "marktwo":
        archive_dir = os.path.expanduser("~/navrep/datasets/V/marktwo")
        if args.dry_run:
            archive_dir = "/tmp/navrep/datasets/V/marktwo"
        env = MarkEnv(silent=True, maps=SECOND_TRAIN_MAPS)
        generate_vae_dataset(
            env, n_sequences=n_sequences,
            subset_index=args.subproc_id, n_subsets=args.n_subprocs,
            render=args.render, archive_dir=archive_dir)
    if args.environment == "navreptrain":
        archive_dir = os.path.expanduser("~/navrep/datasets/V/navreptrain")
        if args.dry_run:
            archive_dir = "/tmp/navrep/datasets/V/navreptrain"
        env = NavRepTrainEnv(silent=True, scenario='train', adaptive=False, collect_statistics=False)
        env.soadrl_sim.human_num = 20
        generate_vae_dataset(
            env, n_sequences=n_sequences,
            subset_index=args.subproc_id, n_subsets=args.n_subprocs,
            policy=ORCAPolicy(suicide_if_stuck=True),
            render=args.render, archive_dir=archive_dir)
    if args.environment == "irl":
        folder_to_archive(
            directory="~/rosbags/iros_rosbags",
            archive_dir=os.path.expanduser("~/navrep/datasets/V/irl"),
        )
