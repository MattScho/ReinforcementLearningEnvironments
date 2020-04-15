from Environments.MouseAndCheese.openMouseAndCheeseEnv import OpenMouseAndCheeseEnv

# Init environment
env = OpenMouseAndCheeseEnv()

print("0. Up\n"
      "1. Down\n"
      "2. Left\n"
      "3. Right\n")
while True:
    # show state
    env.render()
    # get command
    cmd = int(input(">"))
    _, done = env.step(cmd)
    if done:
        print("Congrats")
        break
