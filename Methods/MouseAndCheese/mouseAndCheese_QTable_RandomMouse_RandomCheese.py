'''
Author: Matthew Schofield
Creation Date: 4/12/2020
Version: 4/15/2020
Uses the QTable algorithm to solve a mouse attempting to reach cheese

The mouse is initialized to a random location
The cheese is initialized always to a central location of (5,5)
'''



from Environments.MouseAndCheese.openMouseAndCheeseEnv import OpenMouseAndCheeseEnv
import numpy as np
import random
import math
import pickle as pkl

class MaC_QTable:

    def __init__(self):
        '''
        Creates a Q Table for the Mouse and Cheese environment
        '''
        self.epsilon = 1.0
        self.epsilon_decay = .995
        self.epsilon_min = .1

        self.learningRate = .5
        self.gamma = .5

        self.QTable = np.random.rand(10,10,10,10,4)


    def act(self, state):
        '''
        Request that the model takes an action on the given state
        :param state: state to output an action for
        :return: action to take {0,1,2,4}
        '''
        # Decay epsilon
        self.epsilon *= self.epsilon_decay
        # Check that epsilon is not below its minimum value
        self.epsilon = max(self.epsilon_min, self.epsilon)

        # There is an epsilon chance that a random action will be taken
        if np.random.random() < self.epsilon:
            return [0,1,2,3][np.random.randint(0,4)]
        # Otherwise have the model output a vector with its 'Q' for each state
        pred = self.QTable[state[0]][state[1]][state[2]][state[3]]
        # Output index of largest 'Q', the action to take
        return np.argmax(pred)


    def replay(self, curState, action, reward, newState):
        '''
        Update the QTable based on the event that occured
        :param curState:
        :param action:
        :param reward:
        :param newState:
        :return:
        '''
        currentQ = self.QTable[curState[0]][curState[1]][curState[2]][curState[3]][action]
        self.QTable[curState[0]][curState[1]][curState[2]][curState[3]][action] = currentQ\
            + self.learningRate * (reward + self.gamma * np.max(self.QTable[newState[0]][newState[1]][newState[2]][newState[3]]) -\
                                   currentQ
                                   )


def euclideanDist(pt1, pt2):
    '''
    Calculate the euclidean distance between 2 [x,y] points

    :param pt1: point 1 [x1, y1]
    :param pt2: point 2 [x2, y2]
    :return: int - Euclidean distance between the two points
    '''
    return math.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)

def rewardCalc(score):
    '''
    Adjust the score to a reward

    :return:
    1 if score > 0
    -1 else
    '''
    if score > 0:
        return 1
    else:
        return -1

# Init environment
env = OpenMouseAndCheeseEnv()

# Run 1000 trials, max duration of a trial 100
trials = 1000
trial_len = 100

# Create agent
agent = MaC_QTable()

steps = []
for trial in range(trials):
    # Log current trial
    print(trial)
    # Reinitialize an environment
    env = OpenMouseAndCheeseEnv(
        mouseStartX=random.randint(0,9),
        mouseStartY=random.randint(0,9),
        cheeseStartX=random.randint(0, 9),
        cheeseStartY=random.randint(0, 9)
    )
    # Get current state and shape it to a proper input format for the agent
    curState = [env.getMousePosition()[0], env.getMousePosition()[1], env.getCheesePosition()[0], env.getCheesePosition()[1]]
    # Run trial
    for step in range(trial_len):
        # Agent decides on an action
        action = agent.act(curState)
        # Find the score for the current state
        priorScore = euclideanDist(env.getMousePosition(), env.getCheesePosition())
        # Take action
        state, done = env.step(action)
        # Find the score after the action takes place
        afterScore = euclideanDist(env.getMousePosition(), env.getCheesePosition())
        # Calculate the change in score to decide how good the action was
        deltaScore = priorScore - afterScore
        # Calculate reward based on the score delta
        reward = rewardCalc(deltaScore)
        # If the game was won, give large reward
        if done:
            reward = 10
        # Get new state
        newState = [env.getMousePosition()[0], env.getMousePosition()[1], env.getCheesePosition()[0], env.getCheesePosition()[1]]
        # Save information about the states, action and reward
        agent.replay(curState, action, reward, newState)
        # Update current state
        curState = newState
        # If the game was won, show how many steps occurred
        if done:
            print("\t{steps}".format(steps=step))
            break

'''
After Training see how the agent preforms
'''
pkl.dump(agent.QTable, open("mouseAndCheeseQTable.pkl", 'wb+'))
# Init new environment
input("Ready to see the agent preform in a game?(Press enter)")
for sampleGame in range(5):
    input("Press Enter to begin game " + str(sampleGame))
    env = OpenMouseAndCheeseEnv(
        mouseStartX=random.randint(0,9),
        mouseStartY=random.randint(0,9),
        cheeseStartX=random.randint(0, 9),
        cheeseStartY=random.randint(0, 9))
    print()
    print("----------")
    print(sampleGame)
    print("----------")
    # Attempt 100 steps
    for i in range(100):
        # Get state
        curState = [env.getMousePosition()[0], env.getMousePosition()[1], env.getCheesePosition()[0], env.getCheesePosition()[1]]
        # Decide on action
        action = agent.act(curState)
        # Take decided action
        state, done = env.step(action)
        # Show environment
        env.render()
        print()
        # Stop
        if done:
            break
