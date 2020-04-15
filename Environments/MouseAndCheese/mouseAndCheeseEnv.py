from Environments.MouseAndCheese.mouse import Mouse
from Environments.MouseAndCheese.cheese import Cheese

class MouseAndCheeseEnv:
    '''
    Environment where there is a mouse that wants to get to cheese
    '''
    
    def __init__(self, length, width, mouseStartX, mouseStartY, cheeseStartX, cheeseStartY):
        '''
        Initialize the environment

        :param length: Length of the area
        :param width: Width of the area
        :param mouseStartX: X co-ordinate starting position of the mouse
        :param mouseStartY: Y co-ordinate starting position of the mouse
        :param cheeseStartX: X co-ordinate starting position of the cheese
        :param cheeseStartY: Y co-ordinate starting position of the cheese
        '''
        self.map = self.createMap(length, width)
        self.length = length
        self.width = width
        self.mouse = Mouse(mouseStartX, mouseStartY)
        self.cheese = Cheese(cheeseStartX, cheeseStartY)

    def createMap(self, length, width):
        '''
        Creates an empty map
        :param length: length (horizontal) of map
        :param width: width (vertical) of map
        :return: map a 2D array
        '''
        out = []
        for i in range(length):
            temp = []
            for j in range(width):
                temp.append(0)
            out.append(temp)
        return out

    def checkInbounds(self, x, y, strict=False):
        '''
        Check if a given x and y are within the bounds of the map, with a boolean to signify if the function
        should return false or throw an exception on out of bounds
        :param x: x co-ordinate to check
        :param y: y co-ordinate to check
        :param strict: whether to stop and throw exception or return False on out of bounds scenario
        :return: True - x,y within bounds, False- x or y is out of bounds and strict is set to False
        :except: Exception if strict is True and x,y is out of bounds
        '''
        # Check if co-ordinates are within bounds
        if x >= self.width or x < 0 or y >= self.length or y < 0:
            # Throw exception if the strict setting is On
            if strict:
                raise Exception("{x},{y} is out of bounds for map {width}x{length}".format(
                        x=x,
                        y=y,
                        width=self.width,
                        length=self.length
                    )
                )
            # Otherwise just return False
            else:
                return False
        # The given x,y is within bounds
        return True

    def place(self, x, y, symbol, strict=False):
        '''
        symbol to place

        :param x: x co-ordinate to place object
        :param y: y co-ordinate to place object
        :param symbol: symbol to place
        :return: Boolean if placement was successful
        '''
        if self.checkInbounds(x, y, strict):
            self.map[y][x] = symbol
            return True
        else:
            return False

    def render(self):
        '''
        Print to console the environment
        '''
        for row in self.map:
            print(row)

    def getState(self):
        '''
        Get the current state of the environment
        :return: state of the environment
        '''
        return self.map

    def getMousePosition(self):
        '''
        Get the position of the mouse
        :return: [mouseX, mouseY]
        '''
        return self.mouse.getPosition()

    def getCheesePosition(self):
        '''
        Get the position of the cheese
        :return: [cheeseX, cheeseY]
        '''
        return self.cheese.getPosition()