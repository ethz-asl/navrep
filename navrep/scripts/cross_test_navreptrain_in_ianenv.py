import os
import numpy as np
from tqdm import tqdm
from pyniel.python_tools.path_tools import make_dir_if_not_exists

from navrep.tools.commonargs import parse_common_args
from navrep.envs.encodedenv import EncodedEnv
from navrep.envs.navreptrainencodedenv import NavRepTrainEncoder
from navrep.envs.e2eenv import E2EIANEnv, E2E1DIANEnv
from navrep.scripts.test_navrep import NavRepCPolicy
from navrep.scripts.test_e2e import E2ECPolicy, E2E1DCPolicy

def run_test_episodes(env, policy, render=False, num_episodes=1000):
    progress_bar = tqdm(range(num_episodes), total=num_episodes)
    for i in progress_bar:
        live_success = np.mean(env.episode_statistics["goal_reached"])
        ob = env.reset()
        done = False
        env_time = 0
        while not done:
            action = policy.act(ob)
            ob, _, done, info = env.step(action)
            if render:
                env.render('human')  # , save_to_file=True)
            env_time += env._get_dt()
        progress_bar.set_description("Cross Testing: Episode {} - {:.2f}".format(i, live_success))
        if i % 1000 == 0:
            path = "/tmp/cross_test_X_in_X.csv"
            env.episode_statistics.to_csv(path)
            print("{} written.".format(path))

    S = env.episode_statistics

    all_success_rate = np.mean(S["goal_reached"])
    print("ALL: {:.1f}%".format(100.*all_success_rate))
    print("-----------")

    scenarios = np.array(sorted(list(set(S["scenario"]))))
    for scenario in scenarios:
        scenario_specific = S[S["scenario"] == scenario]
        success_rate = np.mean(scenario_specific["goal_reached"])
        print("{}: {}%".format(scenario, 100.*success_rate))

    return S


# run xtest on all stored controllers for the desired backend/encoding combo
if __name__ == '__main__':
    from stable_baselines import PPO2

    args, _ = parse_common_args()

    # defaults
    if args.n is None:
        args.n = 1000
    collect_trajectories = True

    # find compatible c models
    gym_dir = os.path.expanduser("~/navrep/models/gym")
    candidates = sorted(os.listdir(gym_dir))
    compatible = [name for name in candidates if args.backend + "_" + args.encoding in name]
    if not compatible:
        print(candidates)
        print("Compatible C model for backend and encoding not found in candidates")
        raise ValueError
    print("Compatible C models:")
    for name in compatible:
        print(name)
    print()

    # run cross test for each compatible c model
    progress_bar = tqdm(compatible)
    for c_model_name in progress_bar:
        c_model_info = c_model_name.split("_ckpt.zip")[0]
        c_model_path = os.path.join(gym_dir,c_model_name)
        print("C Model '{}' selected".format(c_model_path))

        if "E2E1D" in args.backend:
            # this is necessary because the pickled file expects a module called custom_policy
            import sys
            import navrep.tools.custom_policy
            sys.modules['custom_policy'] = navrep.tools.custom_policy
            # env and policy
            env = E2E1DIANEnv(silent=True, collect_trajectories=collect_trajectories)
            policy = E2E1DCPolicy(c_model_path)
        elif "E2E" in args.backend:
            # this is necessary because the pickled file expects a module called custom_policy
            import sys
            import navrep.tools.custom_policy
            sys.modules['custom_policy'] = navrep.tools.custom_policy
            # env and policy
            env = E2EIANEnv(silent=True, collect_trajectories=collect_trajectories)
            policy = E2ECPolicy(c_model_path)
        else:
            # create ianenv with soadrl encoder
            encoder = NavRepTrainEncoder(args.backend, args.encoding, gpu=not args.no_gpu)
            env = EncodedEnv(None, None, silent=True, encoder=encoder,
                             collect_trajectories=collect_trajectories)

            # gym model
            c_model = PPO2.load(c_model_path)
            policy = NavRepCPolicy(c_model)

        progress_bar.set_description("C model strain")
        S = run_test_episodes(env, policy, render=args.render, num_episodes=args.n)

        DIR = os.path.expanduser("~/navrep/eval/crosstest")
        if args.dry_run:
            DIR = "/tmp/navrep/eval/crosstest"
        make_dir_if_not_exists(DIR)
        if collect_trajectories:
            NAME = "{}_in_ianenv_x{}.pckl".format(c_model_info, len(S))
            PATH = os.path.join(DIR, NAME)
            S.to_pickle(PATH)
        else:
            NAME = "{}_in_ianenv_x{}.csv".format(c_model_info, len(S))
            PATH = os.path.join(DIR, NAME)
            S.to_csv(PATH)
        print("{} written.".format(PATH))
