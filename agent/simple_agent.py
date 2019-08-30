import numpy as np
import random
import config

class Agent():
    def __init__(self):
        np.random.seed(123123)
        self.env = None
        self.memory = []

        self.ob = None
        self.r = None

    def test(self, env):
        self.env = env
        observation = self.env.reset()
        done = False
        while not done:
            action = np.zeros((config.Game.AgentNum))
            ob, r, done, info = self.env.step(action)
            self.memory.append(sum(r))
        total_reward = sum(self.memory)
        self.memory = []
        return total_reward, info
