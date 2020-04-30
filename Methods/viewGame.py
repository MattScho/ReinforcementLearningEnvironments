
import gym
from stable_baselines import PPO1
from stable_baselines.common.policies import MlpPolicy

from stable_baselines.bench import Monitor

env = gym.make("BipedalWalker-v3")
model = PPO1(MlpPolicy, env,  verbose=1)
print("Starting")
model.learn(total_timesteps=400000)


obs = env.reset()
while True:
    action = env.action_space.sample()
    obs, rewards, done, info = env.step(action)
    env.render()
    if done:
        env.reset()

