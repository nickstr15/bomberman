import os
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os

from .Model import Maverick
from .ManagerFeatures import *

PARAMETERS = 'Test1' #select parameter_set stored in network_parameters/

ACTIONS = ['LEFT', 'RIGHT', 'UP', 'DOWN', 'WAIT', 'BOMB']

def setup(self):
    """
    This is called once when loading each agent.
    Preperation such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.network = Maverick()

    if self.train:
        self.logger.info("Trainiere ein neues Model.")

    else:
        self.logger.info(f"Lade Model '{PARAMETERS}'.")
        filename = os.path.join("network_parameters", f'{PARAMETERS}.pt')
        self.network.load_state_dict(torch.load(filename))
        self.network.eval()
    
    self.pos_saver_feat = []

    self.bomb_timer = 0

    

def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    
    if game_state is None:
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    return state_to_features(self, game_state)
