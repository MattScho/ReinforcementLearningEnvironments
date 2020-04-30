import numpy as np
import random
import gym
import math

from stable_baselines import DQN
from stable_baselines.deepq import MlpPolicy
from gym import spaces

class AgentSearchEnv(gym.Env):
    '''
    '''
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(AgentSearchEnv, self).__init__()
        self.mouseX = random.randint(0,9)
        self.mouseY = random.randint(0,9)
        self.cheeseX = random.randint(0,9)
        self.cheeseY = random.randint(0,9)
        self.prevDist = 0
        self.action_space =spaces.Discrete(4)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(10,10,), dtype=np.float32)
        self.time_steps = 0

    def getActionSpace(self):
        return self.action_space

    def getObservationSpace(self):
        return self.observation_space

    def randomAction(self):
        return self.action_space.sample()

    def step(self, action):
        done =False
        self.time_steps += 1
        move = action
        # Up
        if move == 0:
            self.mouseY -= 1
            if self.mouseY < 0:
                self.mouseY = 0
                done = True
                reward = -10
        # Down
        if move == 1:
            self.mouseY += 1
            if self.mouseY > 9:
                self.mouseY = 9
                done = True
                reward = -1
        # Left
        if move == 2:
            self.mouseX -= 1
            if self.mouseX < 0:
                self.mouseX = 0
                done = True
                reward = -1
        # Right
        if move == 3:
            self.mouseX += 1
            if self.mouseX > 9:
                self.mouseX = 9
                done = True
                reward = -1
        if not done:
            curDist = math.sqrt((self.mouseX-self.cheeseX)**2 + (self.mouseY-self.cheeseY)**2)
            reward = self.prevDist - curDist
            self.prevDist = curDist
            if curDist == 0:
                done = True
                reward = 10

        state = list(np.zeros((10,10)))
        state[self.mouseX][self.mouseY] = 1
        state[self.cheeseX][self.cheeseY] = 2

        return np.array(state, dtype=np.float32), reward, done, {}

    def reset(self):
        self.mouseX = random.randint(0,9)
        self.mouseY = random.randint(0,9)
        self.cheeseX = random.randint(0,9)
        self.cheeseY = random.randint(0,9)

        state = list(np.zeros((10,10)))
        state[self.mouseX][self.mouseY] = 1
        state[self.cheeseX][self.cheeseY] = 2

        return np.array(state, dtype=np.float32)

    def render(self, mode='human'):
        print(self.mouseX)
        print(self.mouseY)
        print(self.cheeseX)
        print(self.cheeseY)

    def close(self):
        pass
