from gym.envs.box2d.car_racing import CarRacing
from tqdm import tqdm
import numpy as np

if __name__ == "__main__":
    env = CarRacing()
    env.reset()

    a = np.zeros(3)
    for _ in tqdm(range(1000000)):
        env.step(a)
#         env.render()
