from Environments.MouseAndCheese.mouse import Mouse
from Environments.MouseAndCheese.cheese import Cheese
import math
from gym import spaces
import numpy as np
import gym
import random


class MouseAndCheeseEnvSimplified(gym.Env):
    '''
    Environment where there is a mouse that wants to get to cheese
    '''
    metadata = {'render.modes': ['human']}

    def __init__(self, length, width, mouseStartX, mouseStartY, cheeseStartX=None, cheeseStartY=None):
        '''
        Initialize the environment

        :param length: Length of the area
        :param width: Width of the area
        :param mouseStartX: X co-ordinate starting position of the mouse
        :param mouseStartY: Y co-ordinate starting position of the mouse
        :param cheeseStartX: X co-ordinate starting position of the cheese
        :param cheeseStartY: Y co-ordinate starting position of the cheese
        '''
        super(MouseAndCheeseEnvSimplified, self).__init__()
        self.length = length
        self.width = width
        self.mouse = Mouse(mouseStartX, mouseStartY)
        if cheeseStartX == None:
            cheeseStartX = random.randint(0,width)
        if cheeseStartY == None:
            cheeseStartY = random.randint(0,length)
        self.cheese = Cheese(cheeseStartX, cheeseStartY)
        self.time_steps = 0


        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(0, 9, shape=(1, 4,), dtype=np.float32)

    def euclidDistance(self):
        return math.sqrt((self.mouse.getX() - self.cheese.getX()) ** 2 + (self.mouse.getY() - self.cheese.getY()) ** 2)

    def render(self):
        '''
        Print to console the environment
        '''
        print(self.mouse.getX())
        print(self.mouse.getY())
        print(self.cheese.getX())
        print(self.cheese.getY())
        print()

    def getState(self):
        '''
        Get the current state of the environment
        :return: state of the environment
        '''
        return np.array([[self.mouse.getX(), self.mouse.getY(), self.cheese.getX(), self.cheese.getY()]])

    def getMousePosition(self):
        '''
        Get the position of the mouse
        :return: [mouseX, mouseY]
        '''
        return self.mouse.getPosition()

    def getCheesePosition(self):
        '''
        Get the position of the cheese
        :return: [cheeseX, cheeseY]
        '''
        return self.cheese.getPosition()

    def getActionSpace(self):
        return self.action_space

    def getObservationSpace(self):
        return self.observation_space

    def randomAction(self):
        return self.action_space.sample()

    def step(self, action):

        previous = self.euclidDistance()
        # Move up
        if action == 0:
            # Check move up possible, if so take action and clean up
            if self.mouse.getY() > 0:
                # Make change to the mouse's state
                self.mouse.setY(self.mouse.getY() - 1)
        # Move Down
        elif action == 1:
            # Check move down possible, if so take action and clean up
            if self.mouse.getY() < self.length:
                # Make change to the mouse's state
                self.mouse.setY(self.mouse.getY() + 1)
        # Move Left
        elif action == 2:
            # Check move left possible, if so take action and clean up
            if self.mouse.getX() > 0:
                # Make change to the mouse's state
                self.mouse.setX(self.mouse.getX() - 1)
        # Move Right
        elif action == 3:
            if self.mouse.getX() < self.width:
                # Make change to the mouse's state
                self.mouse.setX(self.mouse.getX() + 1)
        # Check if the game is won
        done = False
        if self.cheese.getX() == self.mouse.getX() and self.cheese.getY() == self.mouse.getY():
            done = True
            reward = 10
        else:
            reward = previous - self.euclidDistance()
            if reward > 0:
                reward = 1
            else:
                reward = -1

        return np.array([[self.mouse.getX(), self.mouse.getY(), self.cheese.getX(), self.cheese.getY()]], dtype=np.float32), reward, done, {}

    def reset(self):
        self.mouse.setX(random.randint(0, 9))
        self.mouse.setY(random.randint(0, 9))
        self.cheese.setX(random.randint(0, 9))
        self.cheese.setY(random.randint(0, 9))

        return np.array([[self.mouse.getX(), self.mouse.getY(), self.cheese.getX(), self.cheese.getY()]], dtype=np.float32)

    def render(self, mode='human'):
        print(self.mouse.getX())
        print(self.mouse.getY())
        print(self.cheese.getX())
        print(self.cheese.getY())
        print()

    def close(self):
        pass
