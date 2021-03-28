import pickle
import random
from typing import List

import events as e

from collections import deque

import numpy as np
import torch
import torch.optim as optim 
import torch.nn as nn

from .ManagerRewards import reward_from_events, rewards_from_own_events
from .ManagerTraining import generate_eps_greedy_policy, add_experience, get_score, track_game_score, save_parameters, update_network, add_remaining_experience, train_network
from .ManagerFeatures import state_to_features

import copy

#TRAINING PLAN
# i |Episoded | eps        | grates | opp | Name  | done?
# -------------------------------------------------------
# 1 | 200     | 1.0-0.001  | 0.00   | 0   | Test1 |
# 2 | 200     | 0.9-0.001  | 0.20   | 0   | Test2 |
# 3 | 200     | 0.7-0.0001 | 0.40   | 0   | Test3 |
# 4 | 200     | 0.6-0.0001 | 0.60   | 0   | Test4 |
# 5 | 200     | 0.6-0.0001 | 0.75   | 0   | Test5 |

#Hyperparameter for Training
TRAIN_FROM_SCRETCH = True
LOAD = 'end_long_training'
# LOAD = 'end_coin_training_1'
SAVE = 'last_save' 

EPSILON = (0.5,0.05)
LINEAR_CONSTANT_QUOTIENT = 0.9

DISCOUNTING_FACTOR = 0.6
BUFFERSIZE = 2000 
BATCH_SIZE = 120 

LOSS_FUNCTION = nn.MSELoss()
OPTIMIZER = optim.Adam
LEARNING_RATE = 0.001

TRAINING_EPISODES = 20000



def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if not TRAIN_FROM_SCRETCH: #load current parameters
        self.network.load_state_dict(torch.load(f'network_parameters\{LOAD}.pt'))
        self.network.eval()

    self.network.initialize_training(LEARNING_RATE, DISCOUNTING_FACTOR, EPSILON, #setup training
                                        BUFFERSIZE, BATCH_SIZE, 
                                        LOSS_FUNCTION, OPTIMIZER,
                                        TRAINING_EPISODES)

    self.epsilon_arr = generate_eps_greedy_policy(self.network, LINEAR_CONSTANT_QUOTIENT) 
    self.experience_buffer = []

    self.episode_counter = 0
    self.total_episodes = TRAINING_EPISODES

    #store game scores for evaluation
    self.game_score = 0      
    self.game_score_arr = []

    self.new_network = copy.deepcopy(self.network) #train not the currently working network



def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that the agent took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    add_experience(self, old_game_state, self_action, new_game_state, events, 5)
    if len(self.experience_buffer) > 0:
        train_network(self)
    
    self.game_score += get_score(events)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    add_experience(self, last_game_state, last_action, None, events, 5)
    if len(self.experience_buffer) > 0:
        train_network(self)

    update_network(self)
    
    self.game_score += get_score(events)
    track_game_score(self)

    self.episode_counter += 1
    if self.episode_counter % (TRAINING_EPISODES // 100) == 0: #save parameters and the game score array
        save_parameters(self, SAVE)
        save_parameters(self, f"save after {self.episode_counter} iterations")
        np.savetxt(f"game_score_{self.episode_counter}",self.game_score_arr)

    if e.SURVIVED_ROUND in events:
        self.logger.info("Runde überlebt!")





    
    

        



