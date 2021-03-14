import numpy as np
import random

from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 

class Maverick(nn.Module):
    def __init__(self):
        super(Maverick, self).__init__()

        self.number_of_in_features = 8
        self.number_of_actions = 6

        #LAYERS
        self.dense1 = nn.Linear(in_features=self.number_of_in_features, out_features=12)
        self.dense2 = nn.Linear(in_features=12, out_features=21)

        self.dense3 = nn.Linear(in_features=21, out_features=10)
        self.out = nn.Linear(in_features=10, out_features=self.number_of_actions)
        # self.out = nn.Linear(in_features=self.number_of_in_features, out_features=self.number_of_actions)



    def forward(self, x):

        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = F.relu(self.dense3(x))
        out = self.out(x)

        return out

    def initialize_training(self, 
                alpha,
                gamma, 
                epsilon, 
                buffer_size,
                batch_size, 
                loss_function,
                optimizer,
                training_episodes):
        self.gamma = gamma
        self.epsilon_begin = epsilon[0]
        self.epsilon_end = epsilon[1]
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.optimizer = optimizer(self.parameters(), lr=alpha)
        self.loss_function = loss_function
        self.training_episodes = training_episodes









    