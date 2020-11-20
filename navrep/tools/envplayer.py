import numpy as np
from pyglet.window import key
from pyniel.python_tools.timetools import WalltimeRate
import time

SIDE_SPEED = 1.
FRONT_SPEED = 1.
BACK_SPEED = 1.
ROT_SPEED = 0.4


class EnvPlayer(object):
    def __init__(self, env, render_mode="human", step_by_step=False):
        self.env = env
        self.render_mode = render_mode
        self.STEP_BY_STEP = step_by_step
        self.boost = False
        self.run()

    def key_press(self, k, mod):
        if k == 0xFF0D or k == key.ESCAPE:
            self.exit = True
        if k == key.R:
            self.restart = True
        if k in [key.RIGHT, key.E]:
            self.action[2] = -ROT_SPEED
        if k in [key.LEFT, key.Q]:
            self.action[2] = +ROT_SPEED
        if k in [key.UP, key.W]:
            self.action[0] = +FRONT_SPEED
        if k in [key.DOWN, key.S]:
            self.action[0] = -BACK_SPEED
        if k in [key.D]:
            self.action[1] = -SIDE_SPEED
        if k in [key.A]:
            self.action[1] = +SIDE_SPEED
        if k in [key.LSHIFT]:
            self.boost = True
        if k in [key.SPACE]:
            pass
        self.action_key_is_set = True

    def key_release(self, k, mod):  # reverse action of pressed
        if k in [key.RIGHT, key.E] and self.action[2] == -ROT_SPEED:
            self.action[2] = 0
        if k in [key.LEFT, key.Q] and self.action[2] == +ROT_SPEED:
            self.action[2] = 0
        if k in [key.UP, key.W] and self.action[0] == +FRONT_SPEED:
            self.action[0] = 0
        if k in [key.DOWN, key.S] and self.action[0] == -BACK_SPEED:
            self.action[0] = 0
        if k in [key.D] and self.action[1] == -SIDE_SPEED:
            self.action[1] = 0
        if k in [key.A] and self.action[1] == +SIDE_SPEED:
            self.action[1] = 0
        if k in [key.LSHIFT]:
            self.boost = False
        if k in [key.SPACE]:
            pass

    def reset(self):
        self.realtime_rate = WalltimeRate(1.0 / self.env._get_dt())
        self.action = np.array([0.0, 0.0, 0.0])
        self.restart = False
        self.exit = False
        self.action_key_is_set = False
        print("Resetting")
        self.env.reset()
        self.restart = False
        self.env.render(mode=self.render_mode)
        self.env._get_viewer().window.on_key_press = self.key_press
        self.env._get_viewer().window.on_key_release = self.key_release

    def run(self):
        # run interactively ----------------------
        self.reset()

        i = 0
        while not self.exit:
            i += 1
            # synchronize (either with keypresses or walltime)
            if self.STEP_BY_STEP:
                # wait for keypress
                while True:
                    if self.boost:
                        break
                    if not self.action_key_is_set:
                        self.env.render(mode=self.render_mode)
                        time.sleep(0.01)
                    else:
                        self.action_key_is_set = False
                        break
            else:
                if not self.boost:
                    self.realtime_rate.sleep()
            # step once
            obs, rew, done, info = self.env.step(self.action)
            self.env.render(mode=self.render_mode)
            #         impglet
            #         pygage.get_buffer_manager().get_color_buffer().save("/tmp/env{:05}.png".format(i))
            if done or self.restart:
                self.reset()
        self.env.close()
