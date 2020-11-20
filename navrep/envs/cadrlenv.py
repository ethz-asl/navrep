from navrep.envs.navreptrainenv import NavRepTrainEnv

class CADRLEnv(NavRepTrainEnv):
    def __init__(self, silent=False, legacy_mode=False, n_humans=10):
        super(CADRLEnv, self).__init__(
            scenario='train', silent=silent, legacy_mode=legacy_mode, adaptive=False, lidar_legs=False)
        self.soadrl_sim.num_circles = 0
        self.soadrl_sim.num_walls = 0
        self.soadrl_sim.human_num = n_humans

    def _add_border_obstacle(self):
        pass


if __name__ == "__main__":
    from navrep.tools.envplayer import EnvPlayer
    env = CADRLEnv()
    player = EnvPlayer(env)
