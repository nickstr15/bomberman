import random
import numpy as np
import torch 
import copy
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

def add_experience(self, old_game_state, self_action, new_game_state, events, n):
    reset = False
    if self.bomb_timer == 5:
        reset = True
        self.bomb_timer = 0
    old_features = state_to_features(self, old_game_state)
    if reset:
        self.bomb_timer = 5
    if old_features is not None:
        if new_game_state is None:
            new_features = old_features
        else:
            reset = False
            if self.bomb_timer > 0:
                self.bomb_timer -= 1
                reset = True
            new_features = state_to_features(self, new_game_state)
            if reset:
                self.bomb_timer += 1
        reward = reward_from_events(self, events)
        reward += rewards_from_own_events(self, old_game_state, self_action, new_game_state, events)
        action_idx = ACTIONS_IDX[self_action]
        action = torch.zeros(6)
        action[action_idx] = 1

        # n-Step TD learning
        add_remaining_experience(self, events, reward, new_features, n)


        self.experience_buffer.append((old_features, action, reward, new_features, 0))
        number_of_elements_in_buffer = len(self.experience_buffer)
        if number_of_elements_in_buffer > self.network.buffer_size:
            self.experience_buffer.pop(0)


def add_remaining_experience(self, events, new_reward, new_features, n):
    steps_back = min(len(self.experience_buffer), n)
    for i in range(1, steps_back+1):
        old_old_features, action, reward, old_new_features, number_of_additional_rewards = self.experience_buffer[-i]
        reward += (self.network.gamma**i)*new_reward
        number_of_additional_rewards += 1
        assert number_of_additional_rewards == i
        self.experience_buffer[-i] = (old_old_features, action, reward, new_features, number_of_additional_rewards)


def train_network(self):
    '''
    network: the network that gets updated
    experience_buffer: the collected experiences, list of game_episodes
    '''

    new_network = self.new_network  # apply updates to
    old_network = self.network      # used in training, used to calculate Y
    experience_buffer = self.experience_buffer

    #randomly choose batch out of the experience buffer
    number_of_elements_in_buffer = len(experience_buffer)
    batch_size = min(number_of_elements_in_buffer, old_network.batch_size)

    random_i = [random.randrange(number_of_elements_in_buffer) for _ in range(batch_size)]

    #compute for each experience in the batch 
    # - the Ys using n-step TD Q-learning
    # - the current guess for the Q function
    sub_batch = []
    Y = []
    for i in random_i:
        random_experience = experience_buffer[i]
        sub_batch.append(random_experience)
    
    for b in sub_batch:
        old_features = b[0]
        action = b[1]
        reward = b[2]
        new_features = b[3]
        number_of_additional_rewards = b[4]

        y = reward
        if new_features is not None:
            y += old_network.gamma**(number_of_additional_rewards+1) * torch.max(old_network(new_features))

        Y.append(y)

    Y = torch.tensor(Y)

    #Qs
    states = torch.cat(tuple(b[0] for b in sub_batch))  #put all states of the sub_batch in one batch
    q_values = new_network(states)
    actions = torch.cat([b[1].unsqueeze(0) for b in sub_batch])
    Q = torch.sum(q_values*actions, dim=1)
    
    Residuals = torch.abs(Y-Q)
    batch_size = min(len(Residuals), 50)
    _, indices = torch.topk(Residuals, batch_size)

    Y_reduced = Y[indices]
    Q_reduced = Q[indices]

    # loss = new_network.loss_function(Q, Y)
    loss = new_network.loss_function(Q_reduced, Y_reduced)
    new_network.optimizer.zero_grad()
    loss.backward()
    new_network.optimizer.step()


def update_network(self):
    self.network = copy.deepcopy(self.new_network)




def save_parameters(self, string):
    torch.save(self.network.state_dict(), f"network_parameters/{string}.pt")


# track the true score
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

# Plot our gamescore -> helpful to see if our training is working without much time effort
def track_game_score(self, smooth=False):
    if self.back_in_game >= 20:
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
    # ax.set_yticks(range(255)[::10])
    ax.set_yticks(range(255))
    ax.tick_params(labelsize=16)

    ax.plot(x,y,color='gray',linewidth=0.5, alpha=0.7, zorder=0)

    cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["red","darkorange","green"])
    ax.scatter(x,y,c=y,cmap=cmap,s=40, alpha=0.5, zorder=1)
    try:
        plt.savefig('training_progress.png')
    except:
        ...
    plt.close()

