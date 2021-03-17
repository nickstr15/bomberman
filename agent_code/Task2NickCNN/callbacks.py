import os
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .Model import Maverick
from .ManagerFeatures import *
from .ManagerTraining import rule_based_act

PARAMETERS = 'CoinsAndCrates50' #select parameter_set stored in network_parameters/

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
        self.logger.info("Setting up model from scratch.")

    else:
        self.logger.info("Loading model from saved state.")
        self.network.load_state_dict(torch.load(f'network_parameters/{PARAMETERS}.pt'))
        self.network.eval()

    

def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    
    if game_state is None:
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .0, .0])

    if self.train: # Exploration vs exploitation

        eps = self.epsilon_arr[self.episode_counter]
        if random.random() <= eps:
            #if random.random() <= 0.75:
                #return rule_based_act(self, game_state)
            return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1]) #EXPLORATION

    features = state_to_features(game_state)
    Q = self.network(features)
    action_prob = np.array(torch.softmax(Q,dim=1).detach().squeeze())
    prob_good_action = action = np.random.choice(ACTIONS, p=action_prob)
    best_action = ACTIONS[np.argmax(action_prob)]

    #___SOFT DECISION___#
    # self.logger.info("action returned by callbacks#act: " + prob_good_action) 
    # return prob_good_action

    #___HARD DECISION___#
    self.logger.debug("action returned by callbacks#act: " + best_action)
    return best_action
