import os

from navrep.envs.dreamenv import DreamEnv

class MarkTwoDreamEnv(DreamEnv):
    def __init__(self, temperature=0.25):
        super(MarkTwoDreamEnv, self).__init__(
            temperature=temperature,
            initial_z_path=os.path.expanduser(
                "~/navrep/datasets/M/marktwo/000_mus_logvars_robotstates_actions_rewards_dones.npz"
            ),
            rnn_model_path=os.path.expanduser("~/navrep/models/M/marktwornn.json"),
            vae_model_path=os.path.expanduser("~/navrep/models/V/marktwovae.json"),
        )
