from Environments.MouseAndCheese.mouseAndCheeseEnv import MouseAndCheeseEnv
from Environments.MouseAndCheese.mouse import Mouse
from Environments.MouseAndCheese.cheese import Cheese
import random

class OpenMouseAndCheeseEnv(MouseAndCheeseEnv):
    '''
    Open environment where a mouse wants to get cheese

    Actions:
    1 - Move up
    2 - Move down
    3 - Move Left
    4 - Move Right


    Tile Codes:
    0 - open space
    1 - Mouse
    2 - Cheese
    '''
    def __init__(self, length=10, width=10, mouseStartX=0, mouseStartY=0, cheeseStartX=5, cheeseStartY=5):
        super().__init__(length, width, mouseStartX, mouseStartY, cheeseStartX, cheeseStartY)
        # Place mouse
        self.place(self.mouse.getX(), self.mouse.getY(), self.mouse.getCode(), strict=True)
        # Place cheese
        self.place(self.cheese.getX(), self.cheese.getY(), self.cheese.getCode(), strict=True)

    def step(self, action):
        '''
        Actions:
        0 - Move up
        1 - Move down
        2 - Move Left
        3 - Move Right

        :param action:
        :return:
        '''
        action = action[0]
        previous = self.euclidDistance()
        # Move up
        if action == 0:
            # Check move up possible, if so take action and clean up
            if self.place(self.mouse.getX(), self.mouse.getY()-1, self.mouse.getCode()):
                # Clean up on map
                self.place(self.mouse.getX(), self.mouse.getY(), 0)
                # Make change to the mouse's state
                self.mouse.setY(self.mouse.getY() - 1)
        # Move Down
        elif action == 1:
            # Check move down possible, if so take action and clean up
            if self.place(self.mouse.getX(), self.mouse.getY()+1, self.mouse.getCode()):
                # Clean up on map
                self.place(self.mouse.getX(), self.mouse.getY(), 0)
                # Make change to the mouse's state
                self.mouse.setY(self.mouse.getY() + 1)
        # Move Left
        elif action == 2:
            # Check move left possible, if so take action and clean up
            if self.place(self.mouse.getX()-1, self.mouse.getY(), self.mouse.getCode()):
                # Clean up on map
                self.place(self.mouse.getX(), self.mouse.getY(), 0)
                # Make change to the mouse's state
                self.mouse.setX(self.mouse.getX() - 1)
        # Move Right
        elif action == 3:
            # Check move right possible, if so take action and clean up
            if self.place(self.mouse.getX()+1, self.mouse.getY(), self.mouse.getCode()):
                # Clean up on map
                self.place(self.mouse.getX(), self.mouse.getY(), 0)
                # Make change to the mouse's state
                self.mouse.setX(self.mouse.getX()+1)
        # Check if the game is won
        done = False
        if self.cheese.getX() == self.mouse.getX() and self.cheese.getY() == self.mouse.getY():
            done = True
        reward = previous - self.euclidDistance()
        if reward > 0:
            reward = 1
        else:
            reward = -1
        return self.map, reward, done, {}

    def reset(self):

        self.map = self.createMap(self.length, self.width)
        self.length = self.length
        self.width = self.width
        self.mouse = Mouse(random.randint(0,self.width), random.randint(0,self.length))
        self.cheese = Cheese(random.randint(0,self.width), random.randint(0,self.length))
