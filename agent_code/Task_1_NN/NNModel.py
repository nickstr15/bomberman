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
                n = 10,
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


import matplotlib.pyplot as plt

def state_to_features(game_state: dict) -> torch.tensor:
    '''
    converts the game_state dict in a RGB image 
    that is returned as a tensor
    '''
    if game_state is None:
        return None
    red = torch.tensor([1,0,0], dtype=torch.float)       #bomb and explosions
    green = torch.tensor([0,1,0], dtype=torch.float)     #own agents
    blue = torch.tensor([0,0,1], dtype=torch.float)      #other agents
    white = red + green + blue                           #stones
    yellow = red + green                                 #coins
    orange = torch.tensor([1,0.5,0], dtype=torch.float)  #crates

    image = torch.zeros((17,17,3)) #initialize black image

    #mark all stones
    mask = torch.tensor(game_state['field'] == -1)
    image[mask] = white * 0.1

    #mark all crates
    is_crate = torch.tensor(game_state['field'] == 1)
    image[is_crate] = orange #TODO scale with possibility for coins left

    #mark all coins
    for coin in game_state['coins']:
        image[coin] = yellow

    #mark own agent
    agent = game_state['self'][3]
    image[agent] = green

    #mark other agents
    for opponent in game_state['others']:
        agent = opponent[3]
        image[agent] = blue

    #mark all bombs and explosions
    #TODO: take in account when/how long bomb explodes/is active
    for bomb, t in game_state['bombs']:
        image[bomb] = red
    active_bombs = torch.tensor(game_state['explosion_map'] != 0)
    image[active_bombs] = red

    image = image.permute(2,0,1) #convert from (x,y,channels) to (channels,x,y)
    image = image[:,1:-1,1:-1]   #remove "border" of stones 

    image = image[0] + image[1] + image[2] / 3 # to gray scale

    image = F.interpolate(image.unsqueeze(0).unsqueeze(0), size=15*5, mode='area') #change scaling from 1px per tile to 5px per tile
                                                                      #image shape (3,75,75)

    # plt.imshow(np.array(image.squeeze(0).permute(1,2,0)))
    # plt.savefig('test.png')

    return image




def reward_from_events(self, events) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """

    # k: finish the game as fast as possible, j: prevent self kills, i: prevent wrong actions
    game_rewards = {
        e.COIN_COLLECTED: 5,
        e.KILLED_OPPONENT: 5,
        e.MOVED_RIGHT: -0.0001,
        e.MOVED_LEFT: -0.0001,
        e.MOVED_UP: -0.0001,
        e.MOVED_DOWN: -0.0001,
        e.WAITED: -0.0001,
        e.INVALID_ACTION: -0.0001,
        e.BOMB_DROPPED: -0.0001,
        e.KILLED_SELF: -10
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






