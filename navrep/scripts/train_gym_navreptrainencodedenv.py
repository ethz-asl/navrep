from datetime import datetime
import os

from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv

from navrep.envs.navreptrainencodedenv import NavRepTrainEncodedEnv, NavRepTrainEncoder
from navrep.tools.sb_eval_callback import NavrepEvalCallback
from navrep.tools.commonargs import parse_common_args

if __name__ == "__main__":
    args, _ = parse_common_args()

    shared_encoder = NavRepTrainEncoder(args.backend, args.encoding, gpu=not args.no_gpu)
    _Z = shared_encoder._Z
    _H = shared_encoder._H

    DIR = os.path.expanduser("~/navrep/models/gym")
    LOGDIR = os.path.expanduser("~/navrep/logs/gym")
    if args.dry_run:
        DIR = "/tmp/navrep/models/gym"
        LOGDIR = "/tmp/navrep/logs/gym"
    START_TIME = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    ENCODER_ARCH = "_{}_{}_V{}M{}".format(args.backend, args.encoding, _Z, _H)
    LOGNAME = "navreptrainencodedenv_" + START_TIME + "_PPO" + ENCODER_ARCH
    LOGPATH = os.path.join(LOGDIR, LOGNAME + ".csv")
    MODELPATH = os.path.join(DIR, LOGNAME + "_ckpt")
    MODELPATH2 = os.path.join(DIR, "navreptrainencodedenv_latest_PPO_ckpt")
    if not os.path.exists(DIR):
        os.makedirs(DIR)
    if not os.path.exists(LOGDIR):
        os.makedirs(LOGDIR)

    MILLION = 1000000
    TRAIN_STEPS = args.n
    if TRAIN_STEPS is None:
        TRAIN_STEPS = 60 * MILLION

    N_ENVS = 6
    if args.debug:
        if args.backend in ["VAE_LSTM", "VAE1D_LSTM"]:
            # can't share encoder for VAE_LSTM as it contains RNN state internally
            shared_encoder = None
        env = DummyVecEnv([lambda: NavRepTrainEncodedEnv(args.backend, args.encoding,
                                                         silent=True, scenario='train',
                                                         gpu=not args.no_gpu,
                                                         shared_encoder=shared_encoder)]*N_ENVS)
    else:
        env = SubprocVecEnv([lambda: NavRepTrainEncodedEnv(args.backend, args.encoding,
                                                           silent=True, scenario='train',
                                                           gpu=not args.no_gpu)]*N_ENVS,
                            start_method='spawn')
    eval_env = NavRepTrainEncodedEnv(args.backend, args.encoding,
                                     silent=True, scenario='train', gpu=not args.no_gpu)
    def test_env_fn():  # noqa
        return NavRepTrainEncodedEnv(args.backend, args.encoding,
                                     silent=True, scenario='test', gpu=not args.no_gpu)
    cb = NavrepEvalCallback(eval_env, test_env_fn=test_env_fn,
                            logpath=LOGPATH, savepath=MODELPATH, verbose=1)
    model = PPO2(MlpPolicy, env, verbose=0)
    model.learn(total_timesteps=TRAIN_STEPS+1, callback=cb)
    obs = env.reset()

    model.save(MODELPATH)
    model.save(MODELPATH2)
    print("Model '{}' saved".format(MODELPATH))

    del model
    env.close()

    model = PPO2.load(MODELPATH)

    env = NavRepTrainEncodedEnv(args.backend, args.encoding,
                                silent=True, scenario='train')
    obs = env.reset()
    for i in range(512):
        action, _states = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)
        if done:
            env.reset()
#         env.render()

    print("exiting.")
    exit()
