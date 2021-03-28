import torch
import numpy as np
from collections import deque
import bisect

STEP = np.array([[1,0], [-1,0], [0,1], [0,-1]])

def state_to_features(self, game_state: dict) -> np.array:

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
        return torch.tensor([1,1,1,1,0,0,0,0]).float().unsqueeze(0)
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
    history = []
    features = []

    way_to_nearest = []

    for pos in (player_pos + STEP):
        if self.pos_saver_feat.count((pos[0], pos[1])) > 4:
            history.append(-self.pos_saver_feat.count((pos[0], pos[1])))
        else:
            history.append(1)
        first_coin = True
        new_distances = np.empty(len(wanted_fields))
        pos = pos.tolist()

        if pos not in possible_next_pos:
            features = np.append(features, -1)
            way_to_nearest.append(np.inf)
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
            index = np.argwhere((wanted_fields==pos).all(axis=1)) # check if pos is in wanted_fields
            if len(index)!=0:
                new_distances[index] = distance
                if first_coin:
                    way_to_nearest.append(distance)
                    first_coin = False

            assert sum((wanted_fields==pos).all(axis=1)) <= 1
            neighbors = possible_neighbors(pos)
            for node in neighbors:              
                q.append([node, distance+1])

        features = np.append(features, np.sum(1/new_distances))
    hot_one = np.argmax(features)
    features[features>=0]=0
    features[hot_one] = 1
    # print("\n\n")
    # print(f"oben:{features[3]}")
    # print(f"unten:{features[2]}")
    # print(f"links:{features[1]}")
    # print(f"rechts:{features[0]}")
    # a = ["rechts", "links", "unten", "oben"]
    # print(f"--> {a[np.argmax(features)]}")
    features = np.append(features, history)
    
    features = torch.from_numpy(features).float()

    # update pos_saver
    self.pos_saver_feat.append(game_state["self"][3])
    if len(self.pos_saver_feat) > 15:
        self.pos_saver_feat.pop(0)

    return features.unsqueeze(0)



#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&#
#&& CHANNELS &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&#
#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&#

#######################
## CHANNEL 1 - COINS ##
#######################
def three_closest_coins(agent_x, agent_y, game_state_coins):
    
    closest_coins = [None,None,None]
    closest_dists = deque([1000,1001,1002])
    for coin_x, coin_y in game_state_coins:
        dist_new = np.linalg.norm([coin_x - agent_x, coin_y - agent_y])
        if dist_new < closest_dists[-1]:
            bisect.insort(closest_dists, dist_new)
            closest_dists.pop()
            position = closest_dists.index(dist_new)
            closest_coins[position] = (coin_x, coin_y)
    
    coins = torch.zeros(3,4)         # closest, 2nd, 3rd
    for i, c in enumerate(closest_coins):
        if c is not None:
            x,y = c
            if   x - agent_x > 0: coins[i][0] = 1 #coin right
            elif x - agent_x < 0: coins[i][1] = 1 #coin left

            if   y - agent_y > 0: coins[i][2] = 1 #coin down
            elif y - agent_y < 0: coins[i][3] = 1 #coin up
    # coins = torch.zeros(4)
    # for i, c in enumerate(closest_coins):
    #     if c is not None:
    #         x,y = c
    #         if   x - agent_x > 0: coins[0] += 1 #coin right
    #         elif x - agent_x < 0: coins[1] += 1 #coin left

    #         if   y - agent_y > 0: coins[2] += 1 #coin down
    #         elif y - agent_y < 0: coins[3] += 1 #coin up

    return coins.reshape(-1)

def closest_coin(agent_x, agent_y, game_state_coins):
    coins = torch.zeros(4)
    closest_coin = None
    closest_dist = 100
    for coin_x, coin_y in game_state_coins:
        dist = np.linalg.norm([coin_x - agent_x, coin_y - agent_y])
        if dist < closest_dist:
            closest_dist = dist 
            closest_coin = [coin_x, coin_y]

    if closest_coin is not None:
        x, y = closest_coin
        if   x - agent_x > 0: coins[0] = 1
        elif x - agent_x < 0: coins[1] = 1

        if   y - agent_y > 0: coins[2] = 1
        elif y - agent_y < 0: coins[3] = 1

    return coins


##############################
## CHANNEL 2 - WALLS&CRATES ##
##############################
def next_walls_and_crates(agent_x, agent_y, field):
    walls = torch.zeros(4)
    crates = torch.zeros(4)
    next_steps = [[1,0],[-1,0],[0,1],[0,-1]]
    for i, (x,y) in enumerate(next_steps):
        interest = field[agent_x+x,agent_y+y]
        if interest == -1: #means here is a stone
            walls[i] = -1
        if interest == 1:
            crates[i] = 1
    return walls, crates



#######################
## CHANNEL 3 - FIRE  ##
#######################
def bombs_and_explosions(agent_x, agent_y, bombs, explosion_map):
    fire = torch.zeros(4)
    # => bombs
    for (x,y), t in bombs: #5 is minimum distance that is save
        if y == agent_y: #if same row: check that row
            if   ((x - agent_x) > 0 ) and ((x-agent_x) <  5): fire[0] = -1
            elif ((x - agent_x) > 0 ) and ((x-agent_x) > -5): fire[1] = -1

        if x == agent_x: #if same column: check that column
            if   ((y - agent_y) > 0 ) and ((y-agent_y) <  5): fire[2] = -1
            elif ((y - agent_y) > 0 ) and ((y-agent_y) > -5): fire[3] = -1
    # => explosions
    next_steps = [[1,0],[-1,0],[0,-1],[0,1]]
    for i, (x,y) in enumerate(next_steps):
        if explosion_map[agent_x+x,agent_y+y] > 0: #means here is an explosion
            fire[i] = -1

    return fire


# channel 5 -> opponents
# opponents = torch.zeros(3,4)         # closest, 2nd, 3rd
# opponents_sorted = [None,None,None]
# closest_dists = deque([1000,1001,1002])
# for other in game_state['others']:
#     op_xy = other[3]
#     dist_new = np.linalg.norm([op_xy[0] - agent_x, op_xy[1] - agent_y])
#     if dist_new < closest_dists[-1]:
#         bisect.insort(closest_dists, dist_new)
#         closest_dists.pop()
#         position = closest_dists.index(dist_new)
#         opponents_sorted[position] = op_xy

# for i, op in enumerate(opponents_sorted):
#     if op is not None:
#         x,y = op
#         if   x - agent_x > 0: opponents[i][0] = 1 #coin right
#         elif x - agent_x < 0: opponents[i][1] = 1 #coin left

#         if   y - agent_y > 0: opponents[i][2] = 1 #coin down
#         elif y - agent_y < 0: opponents[i][3] = 1 #coin up

# opponents = opponents.reshape(-1)