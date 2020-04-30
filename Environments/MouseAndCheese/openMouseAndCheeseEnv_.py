import gym
from gym import spaces
import numpy as np
from Environments.MouseAndCheese.openMouseAndCheeseEnv import OpenMouseAndCheeseEnv

class OpenMouseAndCheeseEnv_(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(OpenMouseAndCheeseEnv_, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(4)
        # Example for using image as input:
        self.observation_space = spaces.Box(low=0, high=10, shape=(10, 10), dtype=np.uint8)
        self.env = OpenMouseAndCheeseEnv()

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        self.env.reset()

    def render(self, mode='human'):
        self.env.render()

    def close (self):
        pass