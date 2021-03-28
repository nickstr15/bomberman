import os
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from random import shuffle

from .Model import Maverick
from .ManagerFeatures import state_to_features
from .ManagerRuleBased import act_rulebased, initialize_rule_based

import events as e

# PARAMETERS = 'last_save' #select parameter_set stored in network_parameters/
PARAMETERS = 'final_parameters' #select parameter_set stored in network_parameters/

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
    
    initialize_rule_based(self)

    self.bomb_buffer = 0
    

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

    features = state_to_features(self, game_state)
    Q = self.network(features)

    if self.train: # Exploration vs exploitation
        eps = self.epsilon_arr[self.episode_counter]
        if random.random() <= eps: # choose random action
            if eps > 0.1:
                if np.random.randint(10) == 0:    # old: 10 / 100 now: 3/4
                    action = np.random.choice(ACTIONS, p=[.167, .167, .167, .167, .166, .166])
                    self.logger.info(f"Waehle Aktion {action} komplett zufaellig")

                    return action

                else:
                    action = act_rulebased(self)

                    self.logger.info(f"Waehle Aktion {action} nach dem rule based agent.")
                    return action
            else:
                action = np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
                self.logger.info(f"Waehle Aktion {action} komplett zufaellig")

                return action

    action_prob	= np.array(torch.softmax(Q,dim=1).detach().squeeze())
    best_action = ACTIONS[np.argmax(action_prob)]
    # best_action = act_rulebased(self)
    self.logger.info(f"Waehle Aktion {best_action} nach dem Hardmax der Q-Funktion")

    return best_action