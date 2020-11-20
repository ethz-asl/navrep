from navrep.envs.navreptrainenv import NavRepTrainEnv
from navrep.tools.commonargs import parse_common_args
from navrep.scripts.test_navrep import run_test_episodes

class LuciaPolicy(object):
    """ legacy SOADRL policy from lucia's paper, takes in agents state, local map """
    def __init__(self, env):
        self.env = env

    def act(self, obs):
        state, local_map = obs
        return self.env.soadrl_sim.robot.act(state, local_map)


if __name__ == '__main__':
    args, _ = parse_common_args()

    env = NavRepTrainEnv(silent=True, scenario='test', legacy_mode=True)
    policy = LuciaPolicy(env)

    run_test_episodes(env, policy, render=args.render)
