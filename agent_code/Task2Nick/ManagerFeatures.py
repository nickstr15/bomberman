import torch
import numpy as np
from collections import deque
import bisect
import heapq
import random

import time

def state_to_features(game_state: dict) -> torch.tensor:
    '''
    converts the game_state dict in a RGB image 
    that is returned as a tensor
    '''
    if game_state is None:
        return None

    #####
    # 0 #
    #####
    #___extract informations___#
    agent = game_state['self'][3] #get position of the agent
    field = game_state['field']
    is_free_field = field == 0   #get fields that are "free" to move to
    complete_free = np.ones((17,17), dtype=bool)
    #coins 
    coins = game_state['coins']
    #list of all bombs and dangerous zones
    death = []
    bombs = game_state['bombs']
    for (bx,by), bt in bombs:
        unsafe = 4 - bt
        for x in range(-unsafe, unsafe+1):
            death.append((bx+x, by))
        for y in range(-unsafe, unsafe+1):
            death.append((bx, by+y))
    #The best position for a bomb is in a dead end (here the most crates get destroyed)
    good_bomb_positions = [(x,y) for x in range(0,16) for y in range(1,16) 
                            if field[x,y] == 0 and isdeadend(field,x,y, death) ]
    #list of all crates
    crates = [(x,y) for x in range(1,16) for y in range(1,16) if field[x,y]==1]
    
    #list of all other players
    others = [(x,y) for (x,y,_,_,_) in game_state['others']]
    
    #####
    # 1 #
    #####
    #___look for next free fields___#
    next_positions = []
    next_steps = [(1,0), (-1,0), (0,1), (0,-1)]
    for sx,sy in next_steps:
        next_x, next_y = agent[0]+sx, agent[1]+sy
        if (next_x, next_y) in death:
            next_positions.append(-2)
        else:
            next_positions.append(field[next_x, next_y])
    



    #######
    # 2.1 #
    #######
    t0 = time.time()
    #___compute closest targets___#
    best_coin = bfs(is_free_field, agent, coins)
    t1 = time.time()-t0
    if t1 > 0.1:
        print(t1)
        print(len(coins))

    best_bomb_position = bfs(is_free_field, agent, good_bomb_positions)
    if best_bomb_position is None:
        best_bomb_position = bfs(is_free_field, agent, crates, returnParent = True)

    clostest_death_danger = bfs(complete_free, agent, death)
    closest_opponent = bfs(complete_free, agent, others)

    #######
    # 2.2 #
    #######
    #___compute relative distances___#
    distances = []
    helper_flags = [] #flags: if agent is in best position this needs to be "visible"
    for target in (best_coin, best_bomb_position, clostest_death_danger, closest_opponent):
        if target is not None:
            dx = target[0]-agent[0] 
            dy = target[1]-agent[1]
            helper_flags.append(1)
        else:
            dx, dy = 0, 0
            helper_flags.append(0)
        distances.append(dx)
        distances.append(dy)

    

    #####
    # 3 #
    #####
    #___flags___#
    #here we add the missing informations
    flags = [0,0,0]
    if game_state['self'][2]:
        flags[0] = 1  #it is possible to throw a bomb

    if helper_flags[1] == 1: #best bomb position found
        flags[1] = 1

    if helper_flags[2] == 1: #standing on dangerous field
        flags[2] = 1

    #####
    # 4 #
    #####
    #___generate tensor___#
    features = next_positions + distances + flags
    #features = [right, left, down, up,
    #            coin_x, coin_y, best_bomb_x, best_bomb_y, death_x, death_y, opp_x, opp_y
    #            canBomb, inBestBombPosition, onnDeathField]

    return torch.tensor(features, dtype=torch.float).unsqueeze(0)

def bfs(is_free_field, start, targets, returnParent = False):
    '''
    returns clostest position of the target_type of targets
    '''
    if len(targets) == 0:
        return None
    else: #search for closest target in targets that is reachable
        q = deque([start])
        visited = set()
        parents = {start: start}
        # counter = 0
        while len(q) > 0:
            # counter += 1
            # if counter > 100000:
            #     print('SHIT')
            #     return None
            pos = q.popleft()
            visited.add(pos)
            if pos in targets: #found closest next target
                if returnParent:
                    return parents[pos]
                return pos
            
            x,y = pos
            new_positions = [(x+1,y), (x-1,y), (x,y+1), (x,y-1)]
            for new_pos in new_positions:
                try:
                    if is_free_field[new_pos]:
                        if new_pos not in visited:
                            q.append(new_pos)
                            parents[(new_pos)] = pos
                except:
                    print(new_pos)
        return None

    

def isdeadend(field, x,y, death):
    counter = 0 
    if (x,y) in death:
        return False
    for sx,sy in [(1,0), (-1,0), (0,1), (0,-1)]:
        if field[x+sx, y+sy] != 0:
            counter += 1
    if counter > 2:
        return True
    return False
    


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
