import torch
import numpy as np
from collections import deque
import bisect
import .settings as s


STEP = np.array([[1,0], [-1,0], [0,1], [0,-1]])

RELEVANT_BOMB_RADIUS = 5 # If a field further away from marverick than this is affected by a bomb, we dont care (it will be gone once
# we reach it.)

MAX_SEARCHING_DISTANCE = 20 # After this distance we reduce the searching accuracy for faster computations, since the game will
# most likely change a lot before we arrive

# Othervise there would be to many features! The nearest 10 should be the only relvant points -> 40! features, thats a lot
MAX_FOUND_CRATE_POSITIONS = 10

def state_to_features(self, game_state: dict) -> np.array:

    max_len_coins = 9 # only the coins

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
        for direction in step:
            for length in range(1, s.BOMB_POWER+1):
                beam = direction*length + pos
                obj = field[beam[0], beam[1]]
                if obj == -1:
                    break
                if obj == 1:
                    destroyed_crates += 1
        return destroyed_crates


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

    possible_next_pos = possible_neighbors(player_pos)
    
    assert len(coins) == max_len_coins
    
    # create the result arrays
    inv_coins = []
    inv_crate_distances = []
    crate_points = []

    # iterate over the steps in all four directions
    for pos in (player_pos + STEP):

        # create the distance arrays
        coin_distances_after_step = np.empty(max_len_coins)
        crate_distances_after_step = np.empty(MAX_FOUND_CRATE_POSITIONS)

        # create the bomb effectiveness array
        expected_destructions_after_step = np.zeros(MAX_FOUND_CRATE_POSITIONS)

        pos = pos.tolist() # needed for the "in" command to work

        if pos not in possible_next_pos:
            # we are walking against a wall or a crate
            inv_coins = np.append(inv_coins, -1)
            continue
        
        # Initialize the distance arrays, if no way can be found we consider the distance to be infinite
        coin_distances_after_step.fill(np.inf)
        crate_distances_after_step.fill(np.inf)


        # initialization of the search algorithm
        visited = [player_pos.tolist()]
        q = deque()
        q.append([pos, 1])

        # Counter for the crate arrays
        number_of_found_crate_positions = 0

        # condition to quit the search early
        found_one = False

        # analyse the change of the distances of the shortest paths to all coins and crates if we do a STEP
        while len(q) != 0:  #TODO replace by Dijkstra if working
            pos, distance = q.popleft()

            # quit the search early if we found a target and if too much steps are exceeded (relevant if few crates)
            if (distance > MAX_SEARCHING_DISTANCE) and (found_one==True):
                break
            
            # skip allready visited positions
            if pos in visited:
                continue

            # mark the current node as visited
            visited.append(pos)

            # coins
            index_coins = np.argwhere((coins==pos).all(axis=1)) # check if pos is in coins -> we reached a coin
            coin_distances_after_step[index_coins] = distance
            if index_coins.any():
                found_one = True

            # crates
            # consider only the MAX_FOUND_CRATE_POSITIONS positions to reduce the features (relevant if many crates)
            # TODO: maybe request better destructions if some are found and reduce MAX_FOUND_CRATE_POSITONS in exchange
            if number_of_found_crate_positions < MAX_FOUND_CRATE_POSITIONS:
                for possible_crate in (pos + STEP):
                    if field[possible_crate[0], possible_crate[1]] == 1:  # one of the neighboring fields is a crate
                        crate_distances_after_step[number_of_found_crate_positions] = distance
                        expected_destructions_after_step[number_of_found_crate_positions] = bomb_effect(possible_crate)
                        number_of_found_crate_positions += 1
                        found_one = True
                        break


            neighbors = possible_neighbors(pos)

            # visit all neighbors
            for node in neighbors:              
                q.append([node, distance+1])

        # append the sum of the inverse distances to the coins for this direction as a feature
        inv_coins = np.append(inv_coins, np.sum(1/coin_distances_after_step))

        # append the inverse crate distances -> here no sum to keep the relation to the bomb_points
        inv_crate_distances = np.append(inv_crate_distances, 1/crate_distances_after_step)

        # append the destroyed crates as a feature
        crate_points = np.append(crate_points, expected_destructions_after_step)


    # END OF THE SEARCH ALGORITHM -> Collect the features in one feature array

    features = []


    # TODO: maybe we have to change the hot one encoding (eg if there are opponents)
    # encode the movement to the coins in one hot manner to crate features that can be used in a linear model
    hot_one = np.argmax(features[0:4])
    features[features>=0]=0
    features[hot_one] = number_of_coins

    # TODO: append the crates features
    features.append

    # TODO: append the bomb feature

    # TODO: append a feature, which indicates if we have our bomb and if not, when we are going to get it back

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