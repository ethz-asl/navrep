import os

from navrep.envs.dreamenv import DreamEnv

class ToyDreamEnv(DreamEnv):
    def __init__(self, temperature=0.25):
        super(ToyDreamEnv, self).__init__(
            temperature=temperature,
            initial_z_path=os.path.expanduser(
                "~/navrep/datasets/M/toy/000_mus_logvars_robotstates_actions_rewards_dones.npz"
            ),
            rnn_model_path=os.path.expanduser("~/navrep/models/M/toyrnn.json"),
            vae_model_path=os.path.expanduser("~/navrep/models/V/toyvae.json"),
        )
