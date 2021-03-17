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

        self.number_of_actions = 6

        #LAYERS
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3)

        self.dense1 = nn.Linear(in_features=24*4*4, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=self.number_of_actions)


    def forward(self, x):

        x = self.conv1(x)                             # 6x11x11 -> 12x7x7
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=1)  # 12x7x7  -> 12x6x6

        x = self.conv2(x)                             # 12x6x6  -> 24x4x4
        x = F.relu(x)

        x = x.reshape(-1, 24*4*4)
        x = self.dense1(x)
        x = F.relu(x)

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









    