
import gym
from stable_baselines import DDPG, DQN, A2C, PPO1, PPO2, ACER, ACKTR, GAIL, TRPO
from stable_baselines.deepq import MlpPolicy

from stable_baselines.bench import Monitor

env = gym.make('LunarLander-v2')
model = DQN(MlpPolicy, env,  verbose=1)
print("Starting")
model.learn(total_timesteps=100000)


obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
    if done:
        env.reset()

