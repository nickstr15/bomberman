#test comment test comment
import numpy as np
def setup(self):
    pass


def act(self, game_state: dict):
    pos = game_state["self"][3]
    bombs = game_state["bombs"]
    for bomb in bombs:
        print(np.array(bomb[1]))
    self.logger.info('Pick action according to pressed key')
    return game_state['user_input']
