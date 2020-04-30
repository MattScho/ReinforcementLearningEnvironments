from Environments.MouseAndCheese.openMouseAndCheeseEnv import OpenMouseAndCheeseEnv

# Init environment
env = OpenMouseAndCheeseEnv()

print("1. Up\n"
      "2. Down\n"
      "3. Left\n"
      "4. Right\n")
while True:
    # show state
    env.render()
    # get command
    cmd = int(input(">"))
    _, done = env.step(cmd)
    if done:
        print("Congrats")
        break
