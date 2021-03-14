import os
import pickle
import random

import numpy as np
from collections import deque
from . import RLModel

ACTIONS = ['LEFT', 'RIGHT', 'UP', 'DOWN', 'WAIT', 'BOMB']
STEP = np.array([[1,0], [-1,0], [0,1], [0,-1]])


#Hyperparameter
GAMMA = 0.4 # discounting factor
ALPHA = 0.001 # learning rate

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
    number_of_features = 4

    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        # TODO initialize para_vecs properly
        self.para_vecs = np.random.rand(6, number_of_features)  # 6 = number of possible movements

    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.para_vecs = pickle.load(file)
    print(self.para_vecs)
    
    # hand crafted feature vecs

    self.para_vecs = np.zeros((6,number_of_features))
    self.para_vecs[1][0] = 1
    self.para_vecs[0][1] = 1
    self.para_vecs[3][2] = 1
    self.para_vecs[2][3] = 1

    self.model = RLModel.Model(number_of_features, GAMMA, ALPHA, self.para_vecs)
    self.counter = 0


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    self.features = state_to_features(game_state)
    Q, best_action_idx = self.model.predict_action(self.features)
    # todo Exploration vs exploitation

    reduction_factor = 0.01  # How fast reduce the randomness
    random_prob = 0.7 - (self.counter*reduction_factor)
    if self.train:
        if self.counter*reduction_factor > 1:
            random_prob = 0

        if random.random() < random_prob:
            self.logger.debug("Choosing action purely at random.")
            # 80%: walk in any direction. 15% wait. 5% bomb.
            return np.random.choice(ACTIONS, p=[.225, .225, .225, .225, .095, .005])
        
        self.logger.debug("Choosing action using hardmax.")
        return ACTIONS[best_action_idx]
        # self.logger.debug("Choosing action using softmax.")
        # return np.random.choice(ACTIONS, p=action_prop)

    # self.logger.debug("Choosing action using softmax.")
    # print(action_prop)
    # return np.random.choice(ACTIONS, p=action_prop)
    self.logger.debug("Choose action with highest prob.")
    a = ACTIONS[best_action_idx]
    self.logger.debug(f"make move {a}")
    return a

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
        return np.array([1,1,1,1])
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
    return features