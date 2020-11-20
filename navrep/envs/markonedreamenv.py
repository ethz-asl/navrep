import os

from navrep.envs.dreamenv import DreamEnv

class MarkOneDreamEnv(DreamEnv):
    def __init__(self, temperature=0.25):
        super(MarkOneDreamEnv, self).__init__(
            temperature=temperature,
            initial_z_path=os.path.expanduser(
                "~/navrep/datasets/M/markone/000_mus_logvars_robotstates_actions_rewards_dones.npz"
            ),
            rnn_model_path=os.path.expanduser("~/navrep/models/M/markonernn.json"),
            vae_model_path=os.path.expanduser("~/navrep/models/V/markonevae.json"),
        )
