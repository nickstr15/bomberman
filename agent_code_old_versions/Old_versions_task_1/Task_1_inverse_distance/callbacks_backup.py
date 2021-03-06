import os
import pickle
import random

import numpy as np
from . import RLModel

ACTIONS = ['LEFT', 'RIGHT', 'UP', 'DOWN', 'WAIT', 'BOMB']

#Hyperparameter
N = 10 # n-step Q learning
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
    number_of_features = 246

    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        self.para_vecs = np.random.rand(6, number_of_features)  # 6 = number of possible movements

    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.para_vecs = pickle.load(file)

    self.model = RLModel.Model(number_of_features, N, GAMMA, ALPHA, self.para_vecs)
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
    Q, action_prop, best_action_idx = self.model.predict_action(self.features)
    # todo Exploration vs exploitation

    reduction_factor = 1/400 * 0.003  # How fast reduce the randomness
    random_prob = 1. - (self.counter*reduction_factor)
    if self.train:
        if self.counter*reduction_factor > 1:
            random_prob = 0
        self.counter += 1

        if random.random() < random_prob:
            self.logger.debug("Choosing action purely at random.")
            # 80%: walk in any direction. 15% wait. 5% bomb.
            return np.random.choice(ACTIONS, p=[.225, .225, .225, .225, .10, .0])
        
        self.logger.debug("Choosing action using softmax.")
        return np.random.choice(ACTIONS, p=action_prop)

    self.logger.debug("Choose action with highest prob.")
    a = ACTIONS[best_action_idx]
    self.logger.debug(f"make move {a}")
    return a


def state_to_features(game_state: dict) -> np.array:
    features = []
    features.append(game_state["self"][3][0])
    features.append(game_state["self"][3][1])
    for coin in game_state["coins"]:
        features.append(coin[0])
        features.append(coin[1])
    for no_coin in range(9 - len(game_state["coins"])):
        features.append(0)
        features.append(0)

    for x in range(17):
        for y in range (17):
            if game_state["field"][(x,y)] == -1:
                features.append(x)
                features.append(y)

    return features
    
    

# def state_to_features(game_state: dict) -> np.array:
#         """
#         *This is not a required function, but an idea to structure your code.*

#         Converts the game state to the input of your model, i.e.
#         a feature vector.

#         You can find out about the state of the game environment via game_state,
#         which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
#         what it contains.

#         :param game_state:  A dictionary describing the current game board.
#         :return: np.array
#         """
#         # This is the dict before the game begins and after it ends
#         if game_state is None:
#             return None


#         # Distance changes of the coins:
#         coin_distance = []
#         player_pos = game_state["self"][3]
#         for coin_pos in game_state["coins"]:
#             coin_distance.append(np.linalg.norm(coin_pos-(player_pos+np.array([0,1]))))  # Up
#             coin_distance.append(np.linalg.norm(coin_pos-(player_pos+np.array([0,-1])))) # Down
#             coin_distance.append(np.linalg.norm(coin_pos-(player_pos+np.array([1,0]))))  # Right
#             coin_distance.append(np.linalg.norm(coin_pos-(player_pos+np.array([-1,0])))) # Left
#             # coin_real_distance.append(find_shortest_path_length(player_pos, coin_pos))
        
#         # Possible_steps:
#         possible_moves = []
#         certain_death = []
#         for step in np.array([[0,1], [0,-1], [1,0], [-1,0]]):
#             next = player_pos + step
#             possible_moves.append(game_state["field"][next[0],next[1]])
#             death = False
#             if game_state["explosion_map"][next[0],next[1]] != 0:
#                 death = True

#             '''for bomb in game_state["bombs"]:
#                 # only consider the bombs exploding the next turn
#                 if bomb[1] != 1:
#                     continue
#                 # one coordinate has to match
#                 if bomb[0][0] != player_pos[0] and bomb[0][1] != player_pos[1]:
#                     continue
#                 # Check if the bomb would hit the player

#                 for x in range(-3,1):
#                     if (bomb[0] + np.array([x,0]) == player_pos).all():
#                         blocked = False
#                         for x_ in range(x, 1):
#                             check = bomb[0][0] + np.array([x_,0])
#                             if game_state["field"][check[0],check[1]] == -1:
#                                 blocked = True
#                         if not blocked:
#                             death = True
#                             break

#                 for x in range(0,4):
#                     if (bomb[0] + np.array([x,0]) == player_pos).all():
#                         blocked = False
#                         for x_ in range(0, x):
#                             check = bomb[0][0] + np.array([x_,0])
#                             if game_state["field"][check[0],check[1]] == -1:
#                                 blocked = True
#                         if not blocked:
#                             death = True

#                 for y in range(-3,1):
#                     if (bomb[0] + np.array([0,y]) == player_pos).all():
#                         blocked = False
#                         for y_ in range(x, 1):
#                             check = bomb[0][0] + np.array([0,y_])
#                             if game_state["field"][check[0],check[1]] == -1:
#                                 blocked = True
#                         if not blocked:
#                             death = True
#                             break

#                 for y in range(0,4):
#                     if (bomb[0] + np.array([0,y]) == player_pos).all():
#                         blocked = False
#                         for y_ in range(0, x):
#                             check = bomb[0][0] + np.array([0,y_])
#                             if game_state["field"][check[0],check[1]] == -1:
#                                 blocked = True
#                         if not blocked:
#                             death = True
#             '''
#             certain_death.append(death)
        

#         collected_coins = 9 - len(game_state["coins"])
#         features = np.append(coin_distance, np.zeros(collected_coins*4))
#         features = np.append(features, possible_moves)



#         # # For example, you could construct several channels of equal shape, ...
#         # channels = []
#         # channels.append(...)
#         # # concatenate them as a feature tensor (they must have the same shape), ...
#         # stacked_channels = np.stack(channels)
#         # # and return them as a vector
#         # return stacked_channels.reshape(-1) 
#         return features

# def find_shortest_path_length(start, end):
#     possible_moves = [1,1,1,1] # Left, Right, Up, Down
