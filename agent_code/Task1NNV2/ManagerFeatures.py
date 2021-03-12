import torch
import numpy as np
from collections import deque
import bisect

def state_to_features(game_state: dict) -> torch.tensor:
    '''
    converts the game_state dict in a RGB image 
    that is returned as a tensor
    '''
    if game_state is None:
        return None

    agent_x, agent_y = game_state['self'][3]

    #____Channel-01-Coins___#
    # coins = three_closest_coins(agent_x, agent_y, game_state['coins'])
    coins = closest_coin(agent_x, agent_y, game_state['coins'])
            
    #____Channel-02-Walls___#
    walls, crates = next_walls_and_crates(agent_x, agent_y, game_state['field'])
                
    #____Channel-03-Bombs&Explosions___#
    fire = bombs_and_explosions(agent_x, agent_y, game_state['bombs'], game_state['explosion_map'])

    features = torch.cat((coins,walls,fire)) #len = 4 + 4 + 4 

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
        if   ((x - agent_x) > 0 ) and ((x-agent_x) <  5): fire[0] = -1
        elif ((x - agent_x) > 0 ) and ((x-agent_x) > -5): fire[1] = -1

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