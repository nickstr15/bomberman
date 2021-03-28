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

    #____Channel-00-Moves___#
    moves = get_possible_moves(agent_x, agent_y, game_state)

    #____Channel-01-Coins___#
    coins = closest_coin(agent_x, agent_y, game_state['coins'])
            
    #____Channel-02-Walls___#
    walls_and_crates = next_walls_and_crates(agent_x, agent_y, game_state['field'])
                
    #____Channel-03-Bombs&Explosions___#
    fire = bombs_and_explosions(agent_x, agent_y, game_state['bombs'], game_state['explosion_map'])

    features = torch.cat((moves,coins)) #len = 6+2

    return features.unsqueeze(0)



#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&#
#&& CHANNELS &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&#
#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&#

################################
## CHANNEL 0 - POSSIBLE MOVES ##
################################
def get_possible_moves(agent_x, agent_y, game_state):
    result = torch.tensor([0.,0.,0.,0.,1.,0.]) #it is always allowed to wait
    moves = [[1,0], [-1,0], [0,1], [0,-1]]
    for i, (mx,my) in enumerate(moves):
        if game_state['field'][agent_x+mx,agent_y+my] == 0:
            result[i] = 1
    if game_state['self'][2]:
        result[5] = 1
    return result


#######################
## CHANNEL 1 - COINS ##
#######################
def closest_coin(agent_x, agent_y, game_state_coins):
    N_coins = len(game_state_coins)
    coins = torch.tensor([0.,0.,0.,0.]) #number of coins + direction
    if N_coins == 0:
        coins = torch.tensor([0.,0.]) #number of coins + direction
    closest_coin = None
    closest_dist = 100
    for coin_x, coin_y in game_state_coins:
        dist = np.linalg.norm([coin_x - agent_x, coin_y - agent_y])
        if dist < closest_dist:
            closest_dist = dist 
            closest_coin = [coin_x, coin_y]

    return torch.tensor(closest_coin)


##############################
## CHANNEL 2 - WALLS&CRATES ##
##############################
def next_walls_and_crates(agent_x, agent_y, field):
    result = torch.zeros(4)
    next_steps = [[1,0],[-1,0],[0,1],[0,-1]]
    for i, (x,y) in enumerate(next_steps):
        interest = field[agent_x+x,agent_y+y]
        if interest == -1: #means here is a stone
            result[i] = -1
        if interest == 1:
            result[i] = 1
    return result



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