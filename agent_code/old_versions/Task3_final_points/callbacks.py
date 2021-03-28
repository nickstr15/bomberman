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
from .ManagerFeatures import *

import events as e

# PARAMETERS = 'last_save' #select parameter_set stored in network_parameters/
PARAMETERS = 'save after 14000 iterations' #select parameter_set stored in network_parameters/
# PARAMETERS = 'end_long_training' #select parameter_set stored in network_parameters/

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

    
    if self.bomb_timer > 0:
        self.bomb_timer -= 1

    features = state_to_features(self, game_state)
    Q = self.network(features)



    if self.train: # Exploration vs exploitation
        eps = self.epsilon_arr[self.episode_counter]
        if random.random() <= eps: # choose random action
            if eps > 0.1:
                if np.random.randint(10) != -1:    # old: 10 / 100 now: 3/4
                    action = np.random.choice(ACTIONS, p=[.167, .167, .167, .167, .166, .166])
                    self.logger.info(f"Waehle Aktion {action} komplett zufaellig")

                    if action == "BOMB" and self.bomb_timer==0:
                        self.bomb_timer = 5
                    return action

                else:
                    action = act_rulebased(self, features)
                    if action == "BOMB" and self.bomb_timer==0:
                        self.bomb_timer = 5
                    self.logger.info(f"Waehle Aktion {action} nach dem rule based agent.")
                    return action
            else:
                action = np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
                self.logger.info(f"Waehle Aktion {action} komplett zufaellig")

                if action == "BOMB" and self.bomb_timer==0:
                    self.bomb_timer = 5

                return action

    T = 1

    action_prob	= np.array(torch.softmax(Q/T,dim=1).detach().squeeze())
    best_action = ACTIONS[np.argmax(action_prob)]
    # best_action = act_rulebased(self, features)
    self.logger.info(f"Waehle Aktion {best_action} nach dem Hardmax der Q-Funktion")
    if best_action == "BOMB" and self.bomb_timer==0:
        self.bomb_timer = 5
    return best_action


def act_rulebased(self, features):

    ACTIONS = ['RIGHT', 'LEFT', 'DOWN','UP',  'WAIT', 'BOMB']
    
    Q = np.dot(self.action_array, self.features)
    action = ACTIONS[np.argmax(Q)]
    # print()
    # print(np.array([ACTIONS, Q]).T)
    # print(f"--> {action}")
    return action

def initialize_rule_based(self):
    self.action_array = np.zeros((6,23))
    # coins
    self.action_array[0][0] = 100
    self.action_array[1][1] = 100
    self.action_array[2][2] = 100
    self.action_array[3][3] = 100

    # crates
    self.action_array[0][4] = 33
    self.action_array[1][5] = 33
    self.action_array[2][6] = 33
    self.action_array[3][7] = 33

    # bomb here
    self.action_array[5][8] = 36

    # explosion here
    self.action_array[0][9] = 10
    self.action_array[1][9] = 10
    self.action_array[2][9] = 10
    self.action_array[3][9] = 10
    self.action_array[4][9] = -10
    self.action_array[5][9] = -10

    # run away
    self.action_array[0][10] = 300
    self.action_array[1][11] = 300
    self.action_array[2][12] = 300
    self.action_array[3][13] = 300
    self.action_array[4][10] = 300
    self.action_array[4][11] = 300
    self.action_array[4][12] = 300
    self.action_array[4][13] = 300
    self.action_array[5][10] = 300
    self.action_array[5][11] = 300
    self.action_array[5][12] = 300
    self.action_array[5][13] = 300

    # not run in explosion
    self.action_array[0][14] = 400
    self.action_array[1][15] = 400
    self.action_array[2][16] = 400
    self.action_array[3][17] = 400

    # walk to opponent
    self.action_array[0][18] = 75
    self.action_array[1][19] = 75
    self.action_array[2][20] = 75
    self.action_array[3][21] = 75