import os
import numpy as np
from pyniel.python_tools.path_tools import make_dir_if_not_exists
from stable_baselines.ppo2 import PPO2
import rl_agent.common_custom_policies  # noqa

from navrep.tools.commonargs import parse_common_args
from navrep.envs.ianenv import IANEnv
from navrep.scripts.cross_test_navreptrain_in_ianenv import run_test_episodes

# constants. If you change these, think hard about what which assumptions break.
_90 = 90  # guldenring downsampled scan size
_1080 = 1080  # navrep scan size
_540 = 1080 // 2
_8 = 8  # number of guldenrings waypoints

class GuldenringCPolicy():
    def __init__(self):
        self.model = PPO2.load(os.path.expanduser(
            "~/Code/drl_local_planner_ros_stable_baselines/example_agents/ppo2_1_raw_data_cont_0/ppo2_1_raw_data_cont_0.pkl"))  # noqa

    def act(self, obs):
        action, _state = self.model.predict(obs, deterministic=True)
        return action

class GuldenringWrapperForIANEnv(IANEnv):
    def _convert_obs(self, ianenv_obs):
        scan, robotstate = ianenv_obs
        guldenring_obs = np.zeros((1, _90 + _8 * 2, 1))
        # rotate lidar scan so that first ray is at -pi
        rotated_scan = np.zeros_like(scan)
        rotated_scan[:_540] = scan[_540:]
        rotated_scan[_540:] = scan[:_540]
        # 1080-ray to 90-ray: for each angular section we take the min of the returns
        lidar_upsampling = _1080 // _90
        downsampled_scan = rotated_scan.reshape((-1, lidar_upsampling))
        downsampled_scan = np.min(downsampled_scan, axis=1)
        self.last_guldenring_scan = downsampled_scan  # store to visualize later
        guldenring_obs[0, :_90, 0] = downsampled_scan
        # fill in waypoints with current goal
        for n_wpt in range(_8):
            guldenring_obs[0, _90 + n_wpt * 2:_90 + n_wpt * 2 + 2, 0] = robotstate[:2]
        # Discretize to a resolution of 5cm.
        guldenring_obs = np.round(np.divide(guldenring_obs, 0.05))*0.05
        return guldenring_obs

    def _convert_action(self, guldenring_action):
        vx, omega = guldenring_action
        ianenv_action = np.array([vx, 0., omega])
        return ianenv_action

    def step(self, guldenring_action):
        ianenv_action = self._convert_action(guldenring_action)
        ianenv_obs, reward, done, info = super(GuldenringWrapperForIANEnv, self).step(ianenv_action)
        guldenring_obs = self._convert_obs(ianenv_obs)
        return guldenring_obs, reward, done, info

    def reset(self, *args, **kwargs):
        ianenv_obs = super(GuldenringWrapperForIANEnv, self).reset(*args, **kwargs)
        guldenring_obs = self._convert_obs(ianenv_obs)
        return guldenring_obs

    def render(self, *args, **kwargs):
        lidar_angles_downsampled = np.linspace(-np.pi, np.pi, _90) \
            + self.iarlenv.rlenv.virtual_peppers[0].pos[2]
        kwargs["lidar_angles_override"] = lidar_angles_downsampled
        kwargs["lidar_scan_override"] = self.last_guldenring_scan
        return self.iarlenv.render(*args, **kwargs)


if __name__ == '__main__':
    args, _ = parse_common_args()

    if args.n is None:
        args.n = 1000
    collect_trajectories = False

    env = GuldenringWrapperForIANEnv(silent=True, collect_trajectories=collect_trajectories)
    policy = GuldenringCPolicy()

    S = run_test_episodes(env, policy, render=args.render, num_episodes=args.n)

    DIR = os.path.expanduser("~/navrep/eval/crosstest")
    if args.dry_run:
        DIR = "/tmp/navrep/eval/crosstest"
    make_dir_if_not_exists(DIR)

    if collect_trajectories:
        NAME = "guldenring_in_ianenv_{}.pckl".format(len(S))
        PATH = os.path.join(DIR, NAME)
        S.to_pickle(PATH)
    else:
        NAME = "guldenring_in_ianenv_{}.csv".format(len(S))
        PATH = os.path.join(DIR, NAME)
        S.to_csv(PATH)
    print("{} written.".format(PATH))
