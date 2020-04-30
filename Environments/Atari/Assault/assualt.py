from stable_baselines.common.cmd_util import make_atari_env
import gym

class AssualtEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, num_envs=4):
        super(AssualtEnv, self).__init__()

        self.env = make_atari_env('AssaultNoFrameskip-v4', num_env=1, seed=0)
        self.num_envs = num_envs
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = self.env.action_space
        # Example for using image as input:
        self.observation_space = self.env.observation_space

    def step_wait(self):
        return self.env.step_wait()

    def step_async(self, action):
        return self.env.step_async(action)

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        return self.env.reset()

    def render(self, mode='human'):
        return self.env.render(mode=mode)

    def close(self):
        self.env.close()

    def get_action_meanings(self):
        return self.env.unwrapped.get_action_meanings
