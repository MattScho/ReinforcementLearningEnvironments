from Environments.MouseAndCheese.token import EnvToken

class Mouse(EnvToken):

    def __init__(self, startX, startY):
        '''

        :param startX: initial X location mouse
        :param startY: initial Y location of mouse
        '''
        super().__init__(startX, startY, 1, "Mouse")