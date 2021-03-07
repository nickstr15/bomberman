import os
import pickle
import random

import numpy as np


ACTIONS = ['LEFT', 'RIGHT', 'UP', 'DOWN', 'WAIT', 'BOMB']


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

    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")

        number_of_features = 44

        self.para_vecs = np.random.rand(6, number_of_features)  # 6 = number of possible movements

    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.para_vecs = pickle.load(file)
    self.counter = 0


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation

    reduction_factor = 1/400 * 0.01  # How fast reduce the randomness
    random_prob = 0.8 - (self.counter*reduction_factor)
    if self.counter*reduction_factor > 1:
        random_prob = 0
    self.counter += 1

    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 15% wait. 5% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .15, .05])

    self.logger.debug("Querying model for action.")
    weights = np.dot(self.para_vecs, state_to_features(game_state))
    prob = weights / np.sum(weights)
    return np.random.choice(ACTIONS, p=prob)


def state_to_features(game_state: dict) -> np.array:
        """
        *This is not a required function, but an idea to structure your code.*

        Converts the game state to the input of your model, i.e.
        a feature vector.

        You can find out about the state of the game environment via game_state,
        which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
        what it contains.

        :param game_state:  A dictionary describing the current game board.
        :return: np.array
        """
        # This is the dict before the game begins and after it ends
        if game_state is None:
            return None


        # Distance changes of the coins:
        coin_distance = []
        player_pos = game_state["self"][3]
        for coin_pos in game_state["coins"]:
            coin_distance.append(np.linalg.norm(coin_pos-(player_pos+np.array([0,1]))))  # Up
            coin_distance.append(np.linalg.norm(coin_pos-(player_pos+np.array([0,-1])))) # Down
            coin_distance.append(np.linalg.norm(coin_pos-(player_pos+np.array([1,0]))))  # Right
            coin_distance.append(np.linalg.norm(coin_pos-(player_pos+np.array([-1,0])))) # Left
            # coin_real_distance.append(find_shortest_path_length(player_pos, coin_pos))
        
        # Possible_steps:
        possible_moves = []
        certain_death = []
        for step in np.array([[0,1], [0,-1], [1,0], [-1,0]]):
            next = player_pos + step
            possible_moves.append(game_state["field"][next[0],next[1]])
            death = False
            if game_state["explosion_map"][next[0],next[1]] != 0:
                death = True

            # for bomb in game_state["bombs"]:
            #     if bomb[0][0] != player_pos[0] and
            #     hit = False
            #     if bomb[1] == 1 and hit:
            #         death = True


            certain_death.append(death)
        

        collected_coins = 9 - len(game_state["coins"])
        features = np.append(coin_distance, np.zeros(collected_coins*4))
        features = np.append(features, possible_moves)



        # # For example, you could construct several channels of equal shape, ...
        # channels = []
        # channels.append(...)
        # # concatenate them as a feature tensor (they must have the same shape), ...
        # stacked_channels = np.stack(channels)
        # # and return them as a vector
        # return stacked_channels.reshape(-1) 
        return features

# def find_shortest_path_length(start, end):
#     possible_moves = [1,1,1,1] # Left, Right, Up, Down
