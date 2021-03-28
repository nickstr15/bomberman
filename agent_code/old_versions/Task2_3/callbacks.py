import os
import pickle
import random

import numpy as np
from collections import deque
from . import RLModel
import heapq

ACTIONS = ['RIGHT', 'LEFT', 'DOWN','UP',  'WAIT', 'BOMB']
STEP = np.array([[1,0], [-1,0], [0,1], [0,-1]])


#Hyperparameter
GAMMA = 0.4 # discounting hfactor
ALPHA = 0.001 # learning rate




def generate_eps_greedy_policy(N, q):

    N_1 = int(N*q)
    N_2 = N - N_1
    eps1 = np.linspace(0, 0, N_1)
    if N_1 == N:
        return eps1
    eps2 = np.ones(N_2) * 0
    return np.append(eps1, eps2)

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
        self.para_vecs = np.zeros((6,18))

        # coins
        self.para_vecs[0][0] = 100
        self.para_vecs[1][1] = 100
        self.para_vecs[2][2] = 100
        self.para_vecs[3][3] = 100

        # crates
        self.para_vecs[0][4] = 23
        self.para_vecs[1][5] = 23
        self.para_vecs[2][6] = 23
        self.para_vecs[3][7] = 23

        # bomb here
        self.para_vecs[5][8] = 30

        # explosion here
        self.para_vecs[0][9] = 10
        self.para_vecs[1][9] = 10
        self.para_vecs[2][9] = 10
        self.para_vecs[3][9] = 10
        self.para_vecs[4][9] = -10
        self.para_vecs[5][9] = -10

        # run away
        self.para_vecs[0][10] = 300
        self.para_vecs[1][11] = 300
        self.para_vecs[2][12] = 300
        self.para_vecs[3][13] = 300

        # not run in explosion
        self.para_vecs[0][14] = 400
        self.para_vecs[1][15] = 400
        self.para_vecs[2][16] = 400
        self.para_vecs[3][17] = 400

    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.para_vecs = pickle.load(file)
    
    self.model = RLModel.Model(number_of_features, GAMMA, ALPHA, self.para_vecs)
    self.counter = 0

    self.eps = generate_eps_greedy_policy(200, 0.7)


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

    random_prob = self.eps[self.counter]  # How fast reduce the randomness
    if self.train:
        if random.random() < random_prob:
            self.logger.debug("Choosing action purely at random.")
            # 80%: walk in any direction. 15% wait. 5% bomb.
            return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
        
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

    STEP = np.array([[1,0], [-1,0], [0,1], [0,-1]])

    DIRECTION = {(1,0):0, (-1,0):1, (0,1):2, (0,-1):3}

    MOVE = ["rechts", "links", "unten", "oben"]

    ACTIONS = ['RIGHT', 'LEFT', 'DOWN','UP',  'WAIT', 'BOMB']
    ACTIONS_INVERSE = {"RIGHT": 0, "LEFT": 1, "DOWN": 2, "UP": 3, "WAIT": 4, "BOMB": 5}

    MAX_SEARCHING_DISTANCE = 20 # After this distance we reduce the searching accuracy for faster computations, since the game will
    # most likely change a lot before we arrive

    # Othervise there would be to many features find the three nearest crates
    MAX_FOUND_CRATE_POSITIONS = 3

    # Othervise there would be to many features find the three nearest dead ends
    MAX_FOUND_DEAD_ENDS = 2

    MAX_CRATES = MAX_FOUND_DEAD_ENDS+MAX_FOUND_CRATE_POSITIONS

    max_len_coins = 9 # only the coins
    max_len_crates = 240

    # at the beginning and the end:

    if game_state is None:
        return None

    # calculate the valid movements
    def possible_neighbors(pos):
        result = []
        for new_pos in (pos + STEP):
            if field[new_pos[0], new_pos[1]] == 0:
                result.append(new_pos.tolist())
        return result


    # calculate the effectivenes of a bomb at position pos
    def bomb_effect(pos):
        # TODO: account for opponents
        destroyed_crates = 0
        for direction in STEP:
            for length in range(1, 4):
                beam = direction*length + pos
                obj = field[beam[0], beam[1]]
                if obj == -1:
                    break
                if (obj == 1) and future_explosion_map[beam[0], beam[1]]==1: # we will ge the crate destroyed
                    destroyed_crates += 1
        return destroyed_crates


    # fill a explosion map with the bombs that are going to explode
    def fill_explosion_map(explosions, bombs, field):
        future_explosion_map = (np.copy(explosions)*-4) + 1 # -3 now exploding, 1 no bomb in reach
        for bomb in bombs:
            pos = np.array(bomb[0])
            timer = bomb[1] - 3 # the smaller, the more dangerous
            field[pos[0], pos[1]] = -2

            for direction in STEP:
                for length in range(1, 4):
                    beam = direction*length + pos
                    obj = field[beam[0], beam[1]]
                    if obj == -1:
                        break
                    if future_explosion_map[beam[0], beam[1]] > timer:
                        future_explosion_map[beam[0], beam[1]] = timer

        return future_explosion_map


    # save the position of maverick as ndarray
    player_pos = np.array(game_state["self"][3])

    # save the known positions of the coins
    coins = np.array(game_state["coins"])
    coins_list = coins.tolist()
    number_of_coins = len(coins)

    # save the positions of the crates
    field = np.array(game_state["field"])
    explosions = np.array(game_state["explosion_map"])
    bombs = game_state["bombs"]

    crates = np.argwhere(field==1)
    number_of_crates = len(crates)
    future_explosion_map = fill_explosion_map(explosions, bombs, field)

    # assert (field[x,y] == game_state["field"][x][y])
    # print(field)

    # in the last gamestate during training we have no more coins
    # thus the algorithm would crash if we did not create a fake coin entry somewhere. Append would not wor othervise
    # 0,0 is always in the border and thus not reachable
    if number_of_coins == 0:
        coins = np.zeros((max_len_coins, 2))

    possible_next_pos = possible_neighbors(player_pos)

    # create the result arrays
    inv_coins = np.zeros(4)
    inv_crate_distances = [[] for _ in range(4)]
    crate_points = [[] for _ in range(4)]

    # create the distance arrays
    coin_distances_after_step = np.empty((4, max_len_coins))
    crate_distances_after_step = np.empty((4, MAX_CRATES))

    # create the bomb effectiveness array
    expected_destructions_after_step = np.zeros((4, MAX_CRATES))

    # Initialize the distance arrays, if no way can be found we consider the distance to be infinite
    coin_distances_after_step.fill(np.inf)
    crate_distances_after_step.fill(np.inf)


    visited = [player_pos.tolist()]
    q = []
    for pos in (player_pos + STEP):
        pos = pos.tolist() # needed for the "in" command to work
        # initialization of the search algorithm
        x = pos[0] - player_pos[0]
        y = pos[1] - player_pos[1]
        heapq.heappush(q, (1, pos, DIRECTION[(x,y)]))

    # Counter for the crate arrays
    number_of_found_crate_positions = np.zeros(4)
    number_of_found_dead_ends = np.zeros(4)
    number_of_found_coins = np.zeros(4)

    # condition to quit the search early
    found_one = False
    skipped = [False, False, False, False]

    # analyse the change of the distances of the shortest paths to all coins and crates if we do a STEP
    while len(q) != 0:
        
        # direction = element of STEP
        distance, pos, direction = heapq.heappop(q)
        # quit the search early if we found a target and if too much steps are exceeded (relevant if few crates)
        if (distance > MAX_SEARCHING_DISTANCE) and (found_one==True):
            break
        
        # skip allready visited positions
        if pos in visited:
            continue

        # mark the current node as visited
        visited.append(pos)

        if distance == 1:
            # Safely blown up
            if future_explosion_map[pos[0], pos[1]]==-2:
                inv_coins[direction] = -2
                crate_points[direction] = np.zeros(MAX_CRATES)
                placebo = np.zeros(MAX_CRATES)
                placebo.fill(-2)
                inv_crate_distances[direction] = placebo

                skipped[direction] = True
                continue

            if pos not in possible_next_pos:
                # we are walking against a wall or a crate
                inv_coins[direction] = -1
                crate_points[direction] = np.zeros(MAX_CRATES)
                placebo = np.zeros(MAX_CRATES)
                placebo.fill(-1)
                inv_crate_distances[direction] = placebo

                skipped[direction] = True
                continue


        # coins
        is_coin = pos in coins_list # check if pos is in coins -> we reached a coin
        if is_coin:
            coin_distances_after_step[direction][int(number_of_found_coins[direction])] = distance
            number_of_found_coins[direction] += 1
            # print(pos, MOVE[direction], distance)
        if is_coin and not found_one:
            found_one = True


        neighbors = possible_neighbors(pos)

        # visit all neighbors
        ways_out = 0
        for node in neighbors:
            ways_out += 1
            if (distance+1)<=3 and (future_explosion_map[node[0], node[1]] != 1):
                # estimate that we will loose a half turns, for each bomb field we cross beacuse of unsafty reasons
                heapq.heappush(q, (distance+0.5, node, direction))
            heapq.heappush(q, (distance+1, node, direction))

        # crates
        if future_explosion_map[pos[0], pos[1]] != 1: # this position is already used -> dont drop a bomb
            continue

        dead_end = False
        if (ways_out == 1) and (number_of_found_dead_ends[direction] < MAX_FOUND_DEAD_ENDS):
            # we found a unused dead end, this should be a good bomb position
            index_crates = int(number_of_found_crate_positions[direction] + number_of_found_dead_ends[direction])
            crate_distances_after_step[direction][index_crates] = distance
            expected_destructions_after_step[direction][index_crates] = bomb_effect(pos)

            dead_end = True
            number_of_found_dead_ends[direction] += 1
            found_one = True

        # consider only the MAX_FOUND_CRATE_POSITIONS positions to reduce the features (relevant if many crates)
        # This crates should be closer but are most likely not as good as the dead ends
        if (number_of_found_crate_positions[direction] < MAX_FOUND_CRATE_POSITIONS) and not dead_end:
            for possible_crate in (pos + STEP):
                if field[possible_crate[0], possible_crate[1]] == 1 and (future_explosion_map[possible_crate[0], possible_crate[1]]==1):
                    # one of the neighboring fields is a free crate
                    index_crates = int(number_of_found_crate_positions[direction] + number_of_found_dead_ends[direction])
                    crate_distances_after_step[direction][index_crates] = distance
                    expected_destructions_after_step[direction][index_crates] = bomb_effect(pos)

                    number_of_found_crate_positions[direction] += 1
                    found_one = True
                    break

    for direction in range(4):
        if skipped[direction]:
            continue

        # append the sum of the inverse distances to the coins for this direction as a feature
        inv_coins[direction] = np.sum(1/coin_distances_after_step[direction]**5)

        # append the inverse crate distances -> here no sum to keep the relation to the bomb_points
        inv_crate_distances[direction] = 1/np.array(crate_distances_after_step[direction])

        # append the destroyed crates as a feature
        crate_points[direction] = np.array(expected_destructions_after_step[direction])

    inv_crate_distances = np.array(inv_crate_distances)

    crate_points = np.array(crate_points)

    # END OF THE SEARCH ALGORITHM -> Collect the features in one feature array

    features = []

    # encode the movement to the coins in one hot manner to crate features that can be used in a linear model
    features = np.append(features, inv_coins)

    # append the crates features
    features = np.append(features, np.max(inv_crate_distances * crate_points, axis=1))

    # is it senceful to drop a bomb here?
    neighboring_chest = False
    if future_explosion_map[player_pos[0], player_pos[1]] == 1:
        for pos in player_pos + STEP:
            if (field[pos[0], pos[1]] == 1) and (future_explosion_map[pos[0], pos[1]] == 1): # free crate
                neighboring_chest = True
    if neighboring_chest:
        bomb_here = bomb_effect(player_pos)
    else:
        bomb_here = -1

    if not game_state["self"][2]:
        bomb_here = -1
    
    features = np.append(features, bomb_here)

    
    # append the negative explosion timer +1 of the current field as a feature, => 0 if no bomb is ticking
    # need to run from this field
    features = np.append(features,-(future_explosion_map[player_pos[0], player_pos[1]]-1))

    # append a running away feature:
    running = np.empty(4)
    running.fill(0)
    if future_explosion_map[player_pos[0], player_pos[1]] == 1:
        features = np.append(features, running)
    else:
        running.fill(-3)
        bomb_locations = []
        for bomb in bombs:
            bomb_locations.append(bomb[0])
        bomb_locations = np.array(bomb_locations)
        # TODO rework if more than one bomb
        pos = bomb_locations[0]
        if pos[0] < player_pos[0]:   # -> pos[1] == player_pos[1]

            if field[player_pos[0]+1, player_pos[1]] == 0:
                running[0] = 1        # -> run away in a straight line

            if field[player_pos[0], player_pos[1]+1] == 0:
                running[2] = 7        # -> run around a corner -> Safe
            if field[player_pos[0], player_pos[1]-1] == 0:
                running[3] = 7        # -> run around a corner -> Safe
                

        if pos[0] > player_pos[0]:   # -> pos[1] == player_pos[1]
            
            if field[player_pos[0]-1, player_pos[1]] == 0:
                running[1] = 1        # -> run away in a straight line

            if field[player_pos[0], player_pos[1]+1] == 0:
                running[2] = 7        # -> run around a corner -> Safe
            if field[player_pos[0], player_pos[1]-1] == 0:
                running[3] = 7        # -> run around a corner -> Safe


        if pos[1] < player_pos[1]:   # -> pos[0] == player_pos[0]

            if field[player_pos[0], player_pos[1]+1] == 0:
                running[2] = 1        # -> run away in a straight line

            if field[player_pos[0]+1, player_pos[1]] == 0:
                running[0] = 7        # -> run around a corner -> Safe
            if field[player_pos[0]-1, player_pos[1]] == 0:
                running[1] = 7        # -> run around a corner -> Safe


        if pos[1] > player_pos[1]:   # -> pos[0] == player_pos[0]

            if field[player_pos[0], player_pos[1]-1] == 0:
                running[3] = 1        # -> run away in a straight line

            if field[player_pos[0]+1, player_pos[1]] == 0:
                running[0] = 7        # -> run around a corner -> Safe
            if field[player_pos[0]-1, player_pos[1]] == 0:
                running[1] = 7        # -> run around a corner -> Safe

        if (pos[0] == player_pos[0]) and (pos[1] == player_pos[1]):
            running.fill(3)

        features = np.append(features, running)

    # append a feature that prevents the agent from running into an explosion -> feature to indicate waiting
    danger = np.zeros(4)
    if future_explosion_map[player_pos[0], player_pos[1]] == 1: # current position is save
        dim = 0
        for pos in player_pos + STEP:
            if future_explosion_map[pos[0], pos[1]] == -3:
                danger[dim] = -1
            dim += 1
    features = np.append(features, danger)
 
    return features

