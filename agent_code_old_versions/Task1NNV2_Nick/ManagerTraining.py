import random
import numpy as np
import torch 

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage.filters import uniform_filter1d

from .ManagerFeatures import state_to_features
from .ManagerRewards import *
    
ACTIONS_IDX = {'LEFT':0, 'RIGHT':1, 'UP':2, 'DOWN':3, 'WAIT':4, 'BOMB':5}

def generate_eps_greedy_policy(network, q):

    N = network.training_episodes
    N_1 = int(N*q)
    N_2 = N - N_1
    eps1 = np.linspace(network.epsilon_begin, network.epsilon_end, N_1)
    if N_1 == N:
        return eps1
    eps2 = np.ones(N_2) * network.epsilon_end
    return np.append(eps1, eps2)

def add_experience(self, old_game_state, self_action, new_game_state, events):
    old_features = state_to_features(old_game_state)
    if old_features is not None:
        if new_game_state is None:
            new_features = old_features
        else:
            new_features = state_to_features(new_game_state)
        reward = reward_from_events(self, events)
        reward += rewards_from_own_events(self, old_game_state, self_action, new_game_state, events)

        action_idx = ACTIONS_IDX[self_action]
        action = torch.zeros(6)
        action[action_idx] = 1

        self.experience_buffer.append((old_features, action, reward, new_features))
        number_of_elements_in_buffer = len(self.experience_buffer)
        if number_of_elements_in_buffer > self.network.buffer_size:
            self.experience_buffer.popleft()

def update_network(self):
    '''
    network: the network that gets updated
    experience_buffer: the collected experiences, list of game_episodes
    '''
    network = self.network 
    experience_buffer = self.experience_buffer

    #randomly choose batch out of the experience buffer
    number_of_elements_in_buffer = len(experience_buffer)
    batch_size = min(number_of_elements_in_buffer, network.batch_size)

    random_i = [ random.randrange(number_of_elements_in_buffer) for _ in range(batch_size)]

    #compute for each experience in the batch 
    # - the Ys using n-step TD Q-learning
    # - the current guess for the Q function
    sub_batch = []
    Y = []
    for i in random_i:
        random_experience = experience_buffer[i]
        sub_batch.append(random_experience)
    
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


def save_parameters(self, string):
    torch.save(self.network.state_dict(), f"network_parameters/{string}.pt")


def get_score(events):
    true_game_rewards = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
    }
    score = 0
    for event in events:
        if event in true_game_rewards:
            score += true_game_rewards[event]
    return score

def track_game_score(self, smooth=False):
    self.game_score_arr.append(self.game_score)
    self.game_score = 0

    #plot scores
    y = self.game_score_arr
    if smooth:
        window_size = self.total_episodes // 25
        if window_size < 1:
            window_size = 1
        y = uniform_filter1d(y, window_size, mode="nearest", output="float")
    x = range(len(y))

    fig, ax = plt.subplots(figsize=(20,10))
    ax.set_title('score during training', fontsize=35, fontweight='bold')
    ax.set_xlabel('episode', fontsize=25, fontweight='bold')
    ax.set_ylabel('points', fontsize=25, fontweight='bold')
    ax.grid(axis='y', alpha=0.2, color='gray', zorder=-1)
    ax.set_yticks([0,1,2,3,4,5,6,7,8,9])
    ax.tick_params(labelsize=16)

    ax.plot(x,y,color='gray',linewidth=0.5, alpha=0.7, zorder=0)

    cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["red","darkorange","green"])
    ax.scatter(x,y,c=y,cmap=cmap,s=40, alpha=0.5, zorder=1)
    
    plt.savefig('network_parameters/training_progress.png')
    plt.close()

