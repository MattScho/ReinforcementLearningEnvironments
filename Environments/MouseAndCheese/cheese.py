from Environments.MouseAndCheese.token import EnvToken


class Cheese(EnvToken):

    def __init__(self, startX, startY):
        '''

        :param startX: initial X location cheese
        :param startY: initial Y location of cheese
        '''
        super().__init__(startX, startY, 2, "Cheese")