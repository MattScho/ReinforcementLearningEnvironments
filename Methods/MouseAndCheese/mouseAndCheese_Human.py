from Environments.MouseAndCheese.mouseAndCheeseEnv import MouseAndCheeseEnv
from Environments.MouseAndCheese.mouseAndCheeseEnvSimplified import MouseAndCheeseEnvSimplified
import random
# Init environment
'''
env = MouseAndCheeseEnv(10, 10,
                        random.randint(0, 9),
                        random.randint(0, 9),
                        random.randint(0, 9),
                        random.randint(0, 9))
'''
env = MouseAndCheeseEnvSimplified(10, 10,
                        random.randint(0, 9),
                        random.randint(0, 9),
                        random.randint(0, 9),
                        random.randint(0, 9))

env.reset()
print("1. Up\n"
      "2. Down\n"
      "3. Left\n"
      "4. Right\n")
while True:
    # show state
    env.render()
    # get command
    cmd = int(input(">"))
    state, reward, done, info = env.step(cmd)
    if done:
        print("Congrats")
        break
