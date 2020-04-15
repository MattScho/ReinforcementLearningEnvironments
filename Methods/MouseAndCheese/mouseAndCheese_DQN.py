'''
Author: Matthew Schofield
Creation Date: 4/12/2020
Version: 4/15/2020
Credits:
    Much of the code was inspired by:
        https://towardsdatascience.com/reinforcement-learning-w-keras-openai-dqns-1eed3a5338c
    Ideas inspired by:
        https://arxiv.org/pdf/1312.5602.pdf
'''

from Environments.MouseAndCheese.openMouseAndCheeseEnv import OpenMouseAndCheeseEnv
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
import math
from collections import deque

class MaC_DeepReinforcementNetwork:

    def __init__(self):
        '''
        Creates a Deep Reinforcement Network to solve the Mouse and Cheese environment
        '''
        self.memory = deque(maxlen=10000)

        # Epsilon parameter to handle taking random actions
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.999

        # Future reward discount factor
        self.gamma = .5

        # Create models
        self.model = self.create_model()
        self.target_model = self.create_model()

    def create_model(self):
        '''
        Builds a 2D convolutional model

        :return: model
        '''
        model = Sequential()
        model.add(Conv2D(28, kernel_size=(3,3), input_shape=(10,10,1)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(24, activation="relu"))
        model.add(Dense(4)) # Output 'Q' (though more probabilities) for taking any of the four actions
        model.compile(loss="mse", optimizer='adam')
        return model

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
        pred = self.model.predict(state)
        # Output index of largest 'Q', the action to take
        return np.argmax(pred[0])

    def remember(self, state, action, reward, new_state, done):
        '''
        Store samples of actions

        :param state: previous state
        :param action: action taken
        :param reward: reward for action
        :param new_state: new state
        :param done: whether the game was completed
        '''
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        '''

        :return:
        '''
        # Wait for then use 100 samples
        batch_size = 100
        if len(self.memory) < batch_size:
            return
        # Grab 100 random samples from previous actions and their results
        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            target = self.model.predict(state)
            if done:
                for i in range(4):
                    if action == i:
                        target[0][i] = target[0][i] + reward
                    else:
                        target[0][i] = -1
            else:
                Q_future = max(self.target_model.predict(new_state)[0])
                target[0][action] = reward + Q_future * self.gamma
            self.model.fit(state, target, epochs=1, verbose=0)

    def target_train(self):
        '''
        Update model
        '''
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i]
        self.target_model.set_weights(target_weights)

    def save_model(self, filename):
        '''
        Save model
        :param filename: file to save to
        '''
        self.model.save(filename)

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

# Set whether or not you wish to view the agent train
VIEW_TRAINING = False
# If VIEW_TRAINING is True, then begin viewing at the the trial # below
BEGIN_VIEWING = 10

# Run 100 trials, max duration of a trial 100
trials = 100
trial_len = 100

# Create agent
agent = MaC_DeepReinforcementNetwork()

steps = []
for trial in range(trials):
    # Log current trial
    print(trial)
    # Reinitialize an environment
    env = OpenMouseAndCheeseEnv(
        mouseStartX=random.randint(0,9),
        mouseStartY=random.randint(0,9),
        cheeseStartX=random.randint(0,9),
        cheeseStartY=random.randint(0,9))
    # Get current state and shape it to a proper input format for the agent
    curState = np.array([np.array(env.getState()).reshape((10,10,1))])
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

        # Set this to true to view the agent train
        if VIEW_TRAINING and trial > BEGIN_VIEWING:
            print()
            env.render()
            print("Action Taken: " + str(action))
            print("Reward: " + str(reward))


        # If the game was won, give large reward
        if done:
            reward = 10
        # Get new state
        newState = np.array([np.array([env.getState()]).reshape((10,10,1))])
        # Save information about the states, action and reward
        agent.remember(curState, action, reward, newState, done)
        # Have the model train
        agent.replay()
        agent.target_train()
        # Update current state
        curState = newState
        # If the game was won, show how many steps occurred
        if done:
            print("\t{steps}".format(steps=step))
            break

agent.save_model("success.model")
'''
After Training see how the agent preforms
'''
# Init new environment
env = OpenMouseAndCheeseEnv()

# Attempt 100 steps
for i in range(100):
    # Get state
    curState = np.array([np.array([env.getState()]).reshape((10,10,1))])
    # Decide on action
    action = agent.act(curState)
    # Take decided action
    state, done = env.step(action)
    # Show environment
    env.render()
    # Stop
    if done:
        break
