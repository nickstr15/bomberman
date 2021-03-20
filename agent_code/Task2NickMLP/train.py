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
from .ManagerTraining import *
from .ManagerFeatures import state_to_features

#TRAINING PLAN
# i |Episoded | eps        | grates | opp | Name    | done?
# -------------------------------------------------------
# 1 | 200     | 1.0-0.001  | 0.00   | 0   | Train00 |
# 2 | 200     | 0.9-0.001  | 0.20   | 0   | Test2   |
# 3 | 200     | 0.7-0.0001 | 0.40   | 0   | Test3   |
# 4 | 200     | 0.6-0.0001 | 0.60   | 0   | Test4   |
# 5 | 200     | 0.6-0.0001 | 0.75   | 0   | Test5   |

#Hyperparameter for Training
TRAIN_FROM_SCRETCH = False
LOAD = 'Train75_4000' #not needed if TRAIN_FROM_SCRETCH = True
SAVE = 'Train75_Long' 

EPSILON = (0.3,0.001)
LINEAR_CONSTANT_QUOTIENT = 0.95

DISCOUNTING_FACTOR = 0.6
BUFFERSIZE = 2000 
BATCH_SIZE = 100

LOSS_FUNCTION = nn.MSELoss()
OPTIMIZER = optim.Adam
LEARNING_RATE = 0.001

TRAINING_EPISODES = 50000



def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    rule_based_setup(self)

    if not TRAIN_FROM_SCRETCH:
        self.network.load_state_dict(torch.load(f'network_parameters/{LOAD}.pt'))
        self.network.eval()

    self.network.initialize_training(LEARNING_RATE, DISCOUNTING_FACTOR, EPSILON, 
                                        BUFFERSIZE, BATCH_SIZE, 
                                        LOSS_FUNCTION, OPTIMIZER,
                                        TRAINING_EPISODES)

    self.epsilon_arr = generate_eps_greedy_policy(self.network, LINEAR_CONSTANT_QUOTIENT)
    self.experience_buffer = deque()

    self.episode_counter = 0
    self.total_episodes = TRAINING_EPISODES

    self.pos_saver = [] #store the last 10 positions

    self.game_score = 0 
    self.game_score_arr = []
    self.crate_score = 0
    self.crate_score_arr = []

    self.dead_ends = set()

    self.new_network = copy.deepcopy(self.network)



def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step .

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that the agent took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    add_experience(self, old_game_state, self_action, new_game_state, events)
    if len(self.experience_buffer) > 0:
        train_network(self)
    
    self.logger.info('####################')
    self.logger.info(events)
    self.game_score += get_score(events)
    self.crate_score += get_score_crates(old_game_state, new_game_state)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    # add_experience(self, last_game_state, last_action, None, events)
    # if len(self.experience_buffer) > 0:
    #     update_network(self)
    
    self.game_score += get_score(events)

    track_game_score(self)

    add_experience(self, last_game_state, last_action, None, events)
    if len(self.experience_buffer) > 0:
        train_network(self)

    update_network(self)

    self.episode_counter += 1
    if self.episode_counter % (TRAINING_EPISODES // 10) == 0: #save parameters 2 times
        save_parameters(self, SAVE)

    self.dead_ends = set()






    
    

        



