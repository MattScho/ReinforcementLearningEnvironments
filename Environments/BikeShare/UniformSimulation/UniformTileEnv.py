from Environments.BikeShare.UniformSimulation.UniformTile_core import UniformTile_core
import gym
from gym import spaces
import numpy as np

'''
An interface for the uniform bikeshare environment
'''
class UniformTileEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, length, width, actionsPerEpisode):
        self.name = "UniForm_Tile_%sx%s".format({length, width})
        self.env = UniformTile_core(length, width, actionsPerEpisode)
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(6,6,), dtype=np.float32)
        self.time_steps = 0

    '''
    Reset the environment
    '''
    def reset(self):
        return self.env.reset()

    '''
    Return dimensions of the environment
    '''
    def getLW(self):
        return self.env.length, self.env.width

    '''
    Return the environment name
    '''
    def getName(self):
        return self.name

    '''
    Return the state as
    [stateMatrix, nextInterest]
    '''
    def getState(self):
        return self.env.getState()

    '''
    Show the environment
    '''
    def render(self, method='human'):
        self.env.render()

    '''
    have the environment step
    dir is a list of [0,1,2,3] where:
        0 down
        1 up
        2 right
        3 left
    and incentive is the incentive offered for that direction
    
    Example:
    dir: [1,2]
    incentive: [3,5]
    
    would mean that there is an offer of
    3 currency (say dollars) for a user to move up/north
    and
    5 currency for a user to move right/east
    '''
    def step(self, action):
        return self.env.step(action)

    '''
    A list of the unservice ratios
    '''
    def getUnserviceRatios(self):
        return self.env.getUnserviceRatios()

    '''
    A list of expenses
    '''
    def getExpenses(self):
        return self.env.getExpenses()

    '''
    Returns the intended next move as a vector for use in a neural network
    '''
    def getNextMoveAsVector(self):
        return self.env.getNextMoveAsVector()

    def close(self):
        self.reset()

