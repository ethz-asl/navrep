#!/usr/bin/env python
from navrep.envs.toytcndreamenv import ToyTCNDreamEnv
import numpy as np
from pyniel.python_tools.timetools import WalltimeRate

if __name__ == "__main__":
    # setting up the sim ---------------
    denv = ToyTCNDreamEnv(temperature=0.05)
    # run interactively ----------------------
    realtime_rate = WalltimeRate(1.0 / denv.DT)
    action = np.array([0.0, 0.0, 0.0])
    restart = [False]
    exit = [False]
    boost = [False]
    # key fetching
    from pyglet.window import key

    def key_press(k, mod):
        if k == 0xFF0D or k == key.ESCAPE:
            exit[0] = True
        if k == key.R:
            restart[0] = True
        if k in [key.RIGHT, key.D]:
            action[2] = -0.2
        if k in [key.LEFT, key.A]:
            action[2] = +0.2
        if k in [key.UP, key.W]:
            action[0] = +0.5
        if k in [key.DOWN, key.S]:
            action[0] = -0.3
        if k in [key.E]:
            action[1] = -0.4
        if k in [key.Q]:
            action[1] = +0.4
        if k in [key.LSHIFT]:
            boost[0] = True

    def key_release(k, mod):  # reverse action of pressed
        if k in [key.RIGHT, key.D] and action[2] == -0.2:
            action[2] = 0
        if k in [key.LEFT, key.A] and action[2] == +0.2:
            action[2] = 0
        if k in [key.UP, key.W] and action[0] == +0.5:
            action[0] = 0
        if k in [key.DOWN, key.S] and action[0] == -0.3:
            action[0] = 0
        if k in [key.E] and action[1] == -0.4:
            action[1] = 0
        if k in [key.Q] and action[1] == +0.4:
            action[1] = 0
        if k in [key.LSHIFT]:
            boost[0] = False

    denv.render()
    denv.viewer.window.on_key_press = key_press
    denv.viewer.window.on_key_release = key_release
    i = 0
    while not exit[0]:
        i += 1
        obs, rew, done, info = denv.step(action)
        denv.render()
        #         impglet
        #         pygage.get_buffer_manager().get_color_buffer().save("/tmp/denv{:05}.png".format(i))
        if done or restart[0]:
            print("Resetting")
            denv.reset()
            restart[0] = False
        if not boost[0]:
            realtime_rate.sleep()
denv.viewer.close()
