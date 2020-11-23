import os
from stable_baselines import PPO2

from navrep.tools.custom_policy import CustomPolicy, Custom1DPolicy

class E2ECPolicy(object):
    """ thin wrapper for gym policies """
    def __init__(self, model_path=None, model=None):
        if model is not None:
            self.model = model
        else:
            self.model_path = model_path
            if self.model_path is None:
                self.model_path = os.path.expanduser(
                    "~/navrep/models/gym/e2enavreptrainenv_latest_PPO_ckpt")
            self.model = PPO2.load(self.model_path, policy=CustomPolicy)
            print("Model '{}' loaded".format(self.model_path))

    def act(self, obs):
        action, _states = self.model.predict(obs, deterministic=True)
        return action

class E2E1DCPolicy(object):
    """ thin wrapper for gym policies """
    def __init__(self, model_path=None, model=None):
        if model is not None:
            self.model = model
        else:
            self.model_path = model_path
            if self.model_path is None:
                self.model_path = os.path.expanduser(
                    "~/navrep/models/gym/e2e1dnavreptrainenv_latest_PPO_ckpt")
            self.model = PPO2.load(self.model_path, policy=Custom1DPolicy)
            print("Model '{}' loaded".format(self.model_path))

    def act(self, obs):
        action, _states = self.model.predict(obs, deterministic=True)
        return action
