import gym

class LunarEnv(gym.Env):
    '''
    Wrapper for OpenAI gym's Lunar Lander environment

    Objective: Safely descend to a designated area
    Inspiration: “Rocket Trajectory is a classic topic in Optimal Control”
    Scoring:
        Gaining leg contact
        Reducing velocity
        Reducing distance from landing pad
        +100 Successful landing
        -0.3 per main engine action
    Action Set: Integer [1, 3]
        0: Do nothing
        1: Thrust right (in original orientation) engine
        2: Thrust main engine
        3: Thrust left (in original orientation) engine
    *Observation Space: Vector of length 8
        Normalized (to view port) x and y position of rocket
        X and Y Velocity of rocket (normalized to view port and FPS)
        Angle of the rocket
        Angular velocity (normalized to FPS)
        1/0 if left leg is on the ground, likewise for right leg
    '''
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(LunarEnv, self).__init__()

        self.env = gym.make('LunarLander-v2')
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def getActionSpace(self):
        return self.action_space

    def getObservationSpace(self):
        return self.observation_space

    def randomAction(self):
        return self.action_space.sample()

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        return self.env.reset()

    def render(self, mode='human'):
        return self.env.render(mode=mode)

    def close(self):
        self.env.close()