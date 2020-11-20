from datetime import datetime
import os

from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv

from navrep.envs.encodedenv import EncodedEnv
from navrep.tools.sb_eval_callback import NavrepLogCallback
from navrep.tools.commonargs import parse_common_args

if __name__ == "__main__":
    args, _ = parse_common_args()

    DIR = os.path.expanduser("~/navrep/models/gym")
    LOGDIR = os.path.expanduser("~/navrep/logs/gym")
    START_TIME = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    LOGNAME = "encodedenv_" + START_TIME + "_PPO"
    LOGPATH = os.path.join(LOGDIR, LOGNAME + ".csv")
    MODELPATH = os.path.join(DIR, LOGNAME + "_ckpt")
    MODELPATH2 = os.path.join(DIR, "encodedenv_latest_PPO_ckpt")
    if not os.path.exists(DIR):
        os.makedirs(DIR)
    if not os.path.exists(LOGDIR):
        os.makedirs(LOGDIR)

    N_ENVS = 6
    env = DummyVecEnv([lambda: EncodedEnv(args.backend, args.encoding, silent=True)]*N_ENVS)
    cb = NavrepLogCallback(logpath=LOGPATH, savepath=MODELPATH, verbose=1)
    model = PPO2(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=40000000, callback=cb)
    obs = env.reset()

    model.save(MODELPATH)
    model.save(MODELPATH2)
    print("Model '{}' saved".format(MODELPATH))

    del model

    model = PPO2.load(MODELPATH)

    env = EncodedEnv(args.backend, args.encoding, silent=True)
    obs = env.reset()
    for i in range(512):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)
        if done:
            env.reset()
        env.render()
