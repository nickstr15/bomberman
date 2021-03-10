import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 

import events as e

class DeepQNetwork(nn.Module):
    def __init__(self, 
                alpha=0.01,
                gamma=0.9, 
                epsilon=(1.,0.0001), 
                buffer_size=10000,
                batch_size=100, 
                loss_function = nn.MSELoss(),
                optimizer = optim.SGD):
        super(DeepQNetwork, self).__init__()

        self.number_of_actions = 6
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon_begin = epsilon[0]
        self.epsilon_end = epsilon[1]
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.n = 10

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=30, stride=3) 
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2)

        self.dense1 = nn.Linear(in_features=32*7*7, out_features=256)

        self.out = nn.Linear(in_features=256, out_features=self.number_of_actions)


    def forward(self, x):
        #(1) input layer: x = x

        #(2) hidden conv layer
        x = self.conv1(x) #shape (75x75) -> (16x16)
        x = F.relu(x)

        #(3) hidden conv layer
        x = self.conv2(x) #shape (16x16) -> (7,7)
        x = F.relu(x)

        #(4) hidden dense layer
        x = x.reshape(-1, 32*7*7)
        x = self.dense1(x)
        x = F.relu(x)

        #(5) output layer
        x = self.out(x)

        return x


def state_to_features(game_state: dict) -> torch.tensor:
    '''
    converts the game_state dict in a RGB image 
    that is returned as a tensor
    '''
    agent_x, agent_y = game_state['self'][3]

    # channel 1 -> coins
    coins = tensor.zeros(9,4)
    for i, (coin_x, coin_y) in enumerate(game_state['coins']):
        distance_x = coins_x - agent_x
        distance_y = coins_y - agent_y
        abs_x = abs(distance_x) 
        abs_y = abs(distance_y)
        if distance_x > 0:
            coins[i][0] = 1 / (abs_x+1)
        else:
            coins[i][1] = 1/ (abs_x+1)
        if distance_y > 0:
            coins[i][0] = 1 / (abs_y+1)
        else:
            coins[i][1] = 1/ (abs_y+1)

    coins = coins[troch.randperm(9)].reshape(-1)

    # channel 2 -> walls
    walls = tensor.zeros(4)
    next_steps = [[1,0],[-1,0],[0,1],[0,-1]]
    for i, (x,y) in enumerate(next_steps):
        if game_state['field'][agent_x+x,agent_y+y] == -1:
            walls[i] = 1
    
    features = torch.cat((coins,walls))

    return features.unsqueeze(0)




def reward_from_events(self, events) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """

    # k: finish the game as fast as possible, j: prevent self kills, i: prevent wrong actions
    game_rewards = {
        e.COIN_COLLECTED: 100,
        e.KILLED_OPPONENT: 5,
        e.MOVED_RIGHT: -0.01,
        e.MOVED_LEFT: -0.01,
        e.MOVED_UP: -0.01,
        e.MOVED_DOWN: -0.01,
        e.WAITED: -0.02,
        e.INVALID_ACTION: -0.03,
        e.BOMB_DROPPED: -0.04,
        e.KILLED_SELF: -20
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


def train(network, experience_buffer_in):
    '''
    network: the network
    experience_buffer: the collected experiences, list of game_episodes
    '''
    experience_buffer = [e for e in experience_buffer_in if len(e) > 0]
    #randomly choose batch out of the experience buffer
    number_of_elements_in_buffer = sum( [len(episode) for episode in experience_buffer])
    batch_size = min(number_of_elements_in_buffer, network.batch_size)

    number_of_episodes = len(experience_buffer)
    random_episodes = [ random.randrange(number_of_episodes) for _ in range(batch_size)]
    sub_batch = []
    for tau in random_episodes:
        try:
            length_of_episode = len(experience_buffer[tau])
            t = random.randrange(length_of_episode)
            random_experience = experience_buffer[tau][t]
            # sub_batch.append(((tau,t), random_experience))
            sub_batch.append(random_experience)
        except:
            print(experience_buffer[tau])
    
    #compute for each expereince in the batch 
    # - the Ys using n-step TD Q-learning
    # - the current guess for the Q function
    Y = []
    # for b in sub_batch:
    #     tau = b[0][0]
    #     t = b[0][1]
    #     experience = b[1]
    #     direct_reward = experience[2]                                                   
    #     y = direct_reward                                           # Y = 1 * R_tau,t

    #     length_of_episode = len(experience_buffer[tau])
    #     max_t = min(t+network.n, length_of_episode-1)
    #     for f,i in enumerate(range(t+1, max_t)):                    #     + sum gamma^i * R_tau,t' 
    #         next_experience = experience_buffer[tau][i]
    #         y+= (network.gamma**(f+1))  * next_experience[2]

    #     if max_t == t+network.n:                                    #     + gamma^n * max_a Q 
    #         last_experience = experience_buffer[tau][t+network.n]
    #         state = last_experience[0]
    #         Q_guess = torch.max(network(state))
    #         y+= (network.gamma**network.n) * Q_guess
    #     Y.append(y)
    for b in sub_batch:
        old_state = b[0]
        action = b[1]
        reward = b[2]
        new_state = b[3]

        y = reward
        if new_state is not None:
            y += network.gamma * torch.max(network(new_state))

        Y.append(y)

    Y = torch.tensor(Y)

    #Qs
    states = torch.cat(tuple(b[0] for b in sub_batch))  #put all states of the sub_batch in one batch
    q_values = network(states)
    actions = torch.cat([b[1].unsqueeze(0) for b in sub_batch])
    Q = torch.sum(q_values*actions, dim=1)
    
    loss = network.loss_function(Q, Y)
    network.optimizer.zero_grad()
    loss.backward()
    network.optimizer.step()


def save_parameters(network, count):
    torch.save(network.state_dict(), f"models/model_{count}.pt")