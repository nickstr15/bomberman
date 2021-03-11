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

#Hyperparameter for Training
EPSILON = (1.0,0.00001)

DISCOUNTING_FACTOR = 0.8
BUFFERSIZE = 1000 #2400
BATCH_SIZE = 10 #300

LOSS_FUNCTION = nn.MSELoss()
OPTIMIZER = optim.Adam
LEARNING_RATE = 0.001

TRAINING_EPISODES = 300

SETUP = 'Test' #set name of file for stored parameters


ACTIONS_IDX = {'LEFT':0, 'RIGHT':1, 'UP':2, 'DOWN':3, 'WAIT':4, 'BOMB':5}

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.network.initialize_training(LEARNING_RATE, DISCOUNTING_FACTOR, EPSILON, 
                                        BUFFERSIZE, BATCH_SIZE, 
                                        LOSS_FUNCTION, OPTIMIZER,
                                        TRAINING_EPISODES)

    self.epsilon_arr = generate_eps_greedy_policy(self.network)
    self.experience_buffer = deque()

    self.episode_counter = 0



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
        update_network(self.network, self.experience_buffer)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died .

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    self.episode_counter += 1

    if self.episode_counter % (TRAINING_EPISODES // 2) == 0: #save parameters 2 times
        string = f'{SETUP}_episode_{self.episode_counter}'
        save_parameters(self.network, string)

def add_experience(self, old_game_state, self_action, new_game_state, events):
    old_state = state_to_features(old_game_state)
    if old_state is not None:
        new_state = state_to_features(new_game_state)
        reward = reward_from_events(self, events)
        reward += rewards_from_own_events(self, old_game_state, self_action, new_game_state)

        action_idx = ACTIONS_IDX[self_action]
        action = torch.zeros(6)
        action[action_idx] = 1

        self.experience_buffer.append((old_state, action, reward, new_state))
        number_of_elements_in_buffer = len(self.experience_buffer)
        if number_of_elements_in_buffer > self.network.buffer_size:
            self.experience_buffer.popleft()




    
    

        



