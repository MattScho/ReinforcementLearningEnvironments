class EnvToken:
    '''
    Abstract parent class to act as the data structure for a 'token' in the environment
    these are objects within the environment
    '''
    def __init__(self, x, y, code, name):
        self.x = x
        self.y = y
        self.code = code
        self.name = name

    '''
    Getters
    '''
    def getX(self):
        return self.x

    def getY(self):
        return self.y

    def getCode(self):
        return self.code

    def getName(self):
        return self.name

    def getPosition(self):
        return [self.x, self.y]

    '''
    Setters
    '''
    def setX(self, newX):
        self.x = newX

    def setY(self, newY):
        self.y = newY
