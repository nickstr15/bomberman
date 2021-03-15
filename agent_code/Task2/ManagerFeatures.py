import torch
import numpy as np
from collections import deque
import bisect

STEP = np.array([[1,0], [-1,0], [0,1], [0,-1]])

RELEVANT_BOMB_RADIUS = 5 # If a field further away from marverick than this is affected by a bomb, we dont care (it will be gone once
# we reach it.)

MAX_SEARCHING_DISTANCE = 20

def state_to_features(self, game_state: dict) -> np.array:

    max_len_coins = 9 # only the coins

    max_number_of_crates = 277

    # at the beginning and the end:

    if game_state is None:
        return None

    # calculate the valid movements
    def possible_neighbors(pos):
        result = []
        for new_pos in (pos + STEP):
            if game_state["field"][new_pos[0], new_pos[1]] == 0:
                result.append(new_pos.tolist())

        return result

    # save the position of maverick as ndarray
    player_pos = np.array(game_state["self"][3])

    # save the known positions of the coins
    coins = np.array(game_state["coins"])
    number_of_coins = len(coins)

    # save the positions of the crates
    field = np.array(game_state["field"])
    crates = np.argwhere(field==1)
    number_of_crates = len(crates)
    # assert (field[x,y] == game_state["field"][x][y])
    # print(field)

    # in the last gamestate during training we have no more coins
    # thus the algorithm would crash if we did not create a fake coin entry somewhere. Append would not wor othervise
    # 0,0 is always in the border and thus not reachable
    if number_of_coins == 0:
        coins = np.zeros((max_len_coins, 2))

    # Same for crates:
    if number_of_crates == 0:
        crates = np.zeros((max_number_of_crates, 2))

    # if the len of wanted fields changes, we receive an error
    # => fill it with not reachable entries (e.g. [0,0]) and shuffle afterward to prevent a bias.
    fake_entries_coins = []
    for _ in range(max_len_coins - len(coins)):
        fake_entries_coins.append([0,0])
    if len(fake_entries_coins) != 0:
        coins = np.append(coins, fake_entries_coins, axis=0)
        np.random.shuffle(coins) # prevent a bias by having the fake entries always on the end.
        # all of the coin fields should have the same influence since the order in game_state is arbitrary
        # not relevant if we use the sum approach...

    # same for crates:
    fake_entries_crates = []
    for _ in range(max_number_of_crates - len(crates)):
        fake_entries_crates.append([0,0])
    if len(fake_entries_crates) != 0:
        crates = np.append(crates, fake_entries_crates, axis=0)
        np.random.shuffle(crates)

    possible_next_pos = possible_neighbors(player_pos)
    features = []

    assert len(coins) == max_len_coins
    assert len(crates) == max_number_of_crates
    
    # iterate over the steps in all four directions
    for pos in (player_pos + STEP):

        # create the distance arrays
        coin_distances_after_step = np.empty(max_len_coins)
        crate_distances_after_step = np.empty(max_number_of_crates)

        pos = pos.tolist() # needed for the "in" command to work

        if pos not in possible_next_pos:
            features = np.append(features, -1)
            continue
        
        # Initialize the distance arrays, if no way can be found we consider the distance to be infinite
        coin_distances_after_step.fill(np.inf)
        crate_distances_after_step.fill(np.inf)
        crate_distances_after_step.fill(np.inf)

        # initialization of the search algorithm
        visited = [player_pos.tolist()]
        q = deque()
        q.append([pos, 1])
        found_one = False
        # analyse the change of the distances of the shortest paths to all coins and crates if we do a STEP
        while len(q) != 0:  #TODO replace by Dijkstra if working
            pos, distance = q.popleft()

            # If we found a possible target and exeded the maximum viewing distance we consider each unseen point as unreachable
            if found_one and distance > MAX_SEARCHING_DISTANCE:
                break

            if pos in visited:
                continue
            visited.append(pos) # mark the current node as visited

            # coins
            index_coins = np.argwhere((coins==pos).all(axis=1)) # check if pos is in coins -> we reached a coin
            if index_coins.any():
                found_one = True
            coin_distances_after_step[index_coins] = distance

            # crates
            for possible_crate in (pos + STEP):
                index_crates = np.argwhere((crates==possible_crate).all(axis=1))
                if index_crates.any() and distance + 1 < crate_distances_after_step[index_crates]:
                    crate_distances_after_step[index_crates] = distance + 1
                    found_one = True
            
            crate_distances_after_step[index_crates] = distance

            neighbors = possible_neighbors(pos)

            # visit all neighbors
            for node in neighbors:              
                q.append([node, distance+1])

        # append the sum of the inverse distances to the coins for this direction as a feature
        features = np.append(features, np.sum(1/coin_distances_after_step))

    # encode the movement to the coins in one hot manner to crate features that can be used in a linear model
    hot_one = np.argmax(features)
    features[features>=0]=0
    features[hot_one] = number_of_coins

    # TODO: append the crates feature

    # TODO: append the bomb feature

    # uncomment to print my personal movement suggestion
    # show_suggested_coin_movement(features)
    
    # append the remaining positive total reward
    features = np.append(features, number_of_coins + 1/50 * number_of_crates)

    # crate a torch tensor that can be returned from the features

    features = torch.from_numpy(features).float()

    return features.unsqueeze(0)


def show_suggested_coin_movement(features):
    print("\n\n")
    print(f"oben:{features[3]}")
    print(f"unten:{features[2]}")
    print(f"links:{features[1]}")
    print(f"rechts:{features[0]}")
    a = ["rechts", "links", "unten", "oben"]
    print(f"--> {a[np.argmax(features)]}")