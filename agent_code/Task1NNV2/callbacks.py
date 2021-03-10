import os
import pickle
import random

import numpy as np
from .Model import DeepQNetwork, state_to_features

import torch
import torch.nn as nn
import torch.optim as optim



ACTIONS = ['LEFT', 'RIGHT', 'UP', 'DOWN', 'WAIT', 'BOMB']
#ACTIONS = ['LEFT', 'RIGHT', 'UP', 'DOWN', 'WAIT']

#Hyperparameter
LEARNING_RATE = 0.01
DISCOUNTING_FACTOR = 0.9
EPSILON = (1.0,0.0001)
BUFFERSIZE = 1000
BATCH_SIZE = 100
LOSS_FUNCTION = nn.MSELoss()
OPTIMIZER = optim.Adam

def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.network = DeepQNetwork(LEARNING_RATE, DISCOUNTING_FACTOR, 
                            EPSILON, BUFFERSIZE, BATCH_SIZE, 
                            LOSS_FUNCTION, OPTIMIZER)
                        
    self.epsilon = np.linspace(self.network.epsilon_begin, self.network.epsilon_end, 1000)
    self.counter = 0

    if self.train:
       self.logger.info("Setting up model from scratch.")
       self.step_count = 0
       self.eps = self.network.epsilon_begin

       self.old_action_prob = np.zeros(6)
       

    else:
        self.logger.info("Loading model from saved state.")
        self.network.load_state_dict(torch.load('models\model_200.pt'))
        self.network.eval()

    

def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    features = state_to_features(game_state)
    if features is None:
        return 'WAIT'
    Q = self.network(features)
    action_prob = np.array(torch.softmax(Q,dim=1).detach().squeeze())
    best_action = ACTIONS[np.argmax(action_prob)]

    if self.train:
        # Exploration vs exploitation
        if self.counter < len(self.epsilon):
            eps = self.epsilon[self.counter]
        else:
            eps = self.network.epsilon_end
        if random.random() <= eps:
            self.logger.debug("Choosing random action.")
            return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, 0.19, .01])

        
        if (self.old_action_prob == action_prob).all():
            print('not changing')

        self.logger.debug("Choosing action using softmax.")
        action = np.random.choice(ACTIONS, p=action_prob)
        self.logger.debug(f"make move {action}")
        return action

    print(action_prob)
    self.logger.debug("Choose action with highest prob.")
    self.logger.debug(f"make move {best_action}")
    #return np.random.choice(ACTIONS, p=action_prob)
    return best_action
    #return act_rule_based(self, game_state)
