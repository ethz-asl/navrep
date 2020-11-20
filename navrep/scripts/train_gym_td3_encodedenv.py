from datetime import datetime
import os
import numpy as np

from stable_baselines import TD3
from stable_baselines.td3.policies import MlpPolicy
from stable_baselines.ddpg.noise import NormalActionNoise

from navrep.envs.encodedenv import EncodedEnv
from navrep.tools.sb_eval_callback import NavrepLogCallback
from navrep.tools.commonargs import parse_common_args

if __name__ == "__main__":
    args, _ = parse_common_args()

    DIR = os.path.expanduser("~/navrep/models/gym")
    LOGDIR = os.path.expanduser("~/navrep/logs/gym")
    START_TIME = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    LOGNAME = "encodedenv_" + START_TIME + "_TD3"
    LOGPATH = os.path.join(LOGDIR, LOGNAME + ".csv")
    MODELPATH = os.path.join(DIR, LOGNAME + "_ckpt")
    MODELPATH2 = os.path.join(DIR, "encodedenv_latest_TD3_ckpt")
    if not os.path.exists(DIR):
        os.makedirs(DIR)
    if not os.path.exists(LOGDIR):
        os.makedirs(LOGDIR)

    env = EncodedEnv(args.backend, args.encoding, silent=True)
    cb = NavrepLogCallback(logpath=LOGPATH, savepath=MODELPATH, verbose=1)

    # The noise objects for TD3
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = TD3(MlpPolicy, env, action_noise=action_noise, verbose=1)
    model.learn(total_timesteps=40000000, callback=cb)
    model.save(MODELPATH)
    model.save(MODELPATH2)

    del model

    model = TD3.load(MODELPATH)

    env = EncodedEnv(args.backend, args.encoding, silent=True)
    obs = env.reset()
    for i in range(512):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)
        if done:
            env.reset()
        env.render()
