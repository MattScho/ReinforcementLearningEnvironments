'''
Human play mode for lunar lander

Author: Matthew Schofield
Version: 4/21/2020
'''

from Environments.Lunar.lunar import LunarEnv

# Start
env = LunarEnv()
obs = env.reset()
print(env.action_space)
print(env.observation_space)
# Play
while True:
    # Enter action
    action = int(input(">"))
    obs, reward, done, info = env.step(action)
    # Show
    env.render(mode='human')

    if done:
        obs = env.reset()