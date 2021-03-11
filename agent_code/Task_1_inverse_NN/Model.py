import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 

import events as e

STEP = np.array([[1,0], [-1,0], [0,1], [0,-1]])

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

        self.number_of_in_features = 4
        self.number_of_actions = 6
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon_begin = epsilon[0]
        self.epsilon_end = epsilon[1]
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.loss_function = loss_function


        #LAYERS

        self.dense1 = nn.Linear(in_features=self.number_of_in_features, out_features=128)
        self.dense2 = nn.Linear(in_features=128, out_features=256)

        self.dense3 = nn.Linear(in_features=256, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=self.number_of_actions)


    def forward(self, x):

        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = F.relu(self.dense3(x))
        out = self.out(x)

        return out


def state_to_features(game_state: dict) -> np.array:

    max_len_wanted_fields = 9 # only the coins

    # at the beginning and the end:

    if game_state is None:
        return None

    def possible_neighbors(pos):
        result = []
        for new_pos in (pos + STEP):
            if game_state["field"][new_pos[0], new_pos[1]] == 0:
                result.append(new_pos.tolist())

        return result

    player_pos = np.array(game_state["self"][3])
    wanted_fields = np.array(game_state["coins"])
    if len(wanted_fields) == 0:
        return torch.tensor([1,1,1,1]).float().unsqueeze(0)
    # if the len of wanted fields changes, we receive an error
    # => fill it with not reachable entries (e.g. [16,16]) and shuffle afterward to prevent a bias.
    fake_entries = []
    for _ in range(max_len_wanted_fields - len(wanted_fields)):
        fake_entries.append([16,16])
    if len(fake_entries) != 0:
        wanted_fields = np.append(wanted_fields, fake_entries, axis=0)
        np.random.shuffle(wanted_fields) # prevent a bias by having the fake entries always on the end.
        # all of the coin fields should have the same influence since the order in game_state is arbitrary

    possible_next_pos = possible_neighbors(player_pos)
    features = []
    for pos in (player_pos + STEP):
        new_distances = np.empty(len(wanted_fields))
        pos = pos.tolist()

        if pos not in possible_next_pos:
            features = np.append(features, -1)
            continue

        new_distances.fill(np.inf) # if no way can be found we consider the distance to be infinite

        # analyse the change of the distances of the shortest paths to all coins if we do a STEP
        visited = [player_pos.tolist()]
        q = deque()
        q.append([pos, 1])

        while len(q) != 0:

            pos, distance = q.popleft()
            if pos in visited:
                continue
            visited.append(pos)

            new_distances[np.argwhere((wanted_fields==pos).all(axis=1))] = distance
            assert sum((wanted_fields==pos).all(axis=1)) <= 1
            neighbors = possible_neighbors(pos)
            for node in neighbors:              
                q.append([node, distance+1])

        features = np.append(features, sum(1/new_distances**3))
    features = torch.from_numpy(features).float()
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
        e.WAITED: -0.04,
        e.INVALID_ACTION: -0.04,
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