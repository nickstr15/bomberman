

import torch
import numpy as np
from collections import deque
import bisect
import heapq
import random
from random import shuffle

import time

ROWS = 17
COLS = 17

def state_to_features(game_state: dict) -> torch.tensor:
    '''
    converts the game_state dict in a RGB image 
    that is returned as a tensor
    '''
    if game_state is None:
        return None

    # 4 channels that represent the action array for
    # 1) get coin
    # 2) destroy crate
    # 3) attack opponent

    ############################
    #___extract informations___#
    ############################
    # 0) field and agent information
    field = game_state['field']
    _, score, canBomb, (x, y) = game_state['self']
    # 1) coins
    coins = game_state['coins']
    # 2) best crate targets
    dead_ends = [(x, y) for x in range(1, 16) for y in range(1, 16) if (field[x, y] == 0)
                 and ([field[x + 1, y], field[x - 1, y], field[x, y + 1], field[x, y - 1]].count(0) == 1)]
    crates = [(x, y) for x in range(1, 16) for y in range(1, 16) if (field[x, y] == 1)]
    # 3) others
    others = [xy for (n, s, b, xy) in game_state['others']]
    # 4) death
    bombs = game_state['bombs']
    bomb_xys = [xy for (xy, t) in bombs]
    bomb_map = np.ones(field.shape) * 5
    for (xb, yb), t in bombs:
        for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i, j] = min(bomb_map[i, j], t)
    

    ##################################
    #___get correct step to target___#
    ##################################
    directions = [(x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    valid_tiles, valid_actions = [], torch.zeros(6)
    nValid = 0
    for d in directions:
        if ((field[d] == 0) and
                (game_state['explosion_map'][d] <= 1) and
                (bomb_map[d] > 0) and
                (not d in others) and
                (not d in bomb_xys)):
            valid_tiles.append(d)
    if (x - 1, y) in valid_tiles: 
        valid_actions[0] = 1 #LEFT
        nValid+=1 
    if (x + 1, y) in valid_tiles: 
        valid_actions[1] = 1 #RIGHT
        nValid+=1 
    if (x, y - 1) in valid_tiles: 
        valid_actions[2] = 1 #UP
        nValid+=1 
    if (x, y + 1) in valid_tiles: 
        valid_actions[3] = 1 #DOWN
        nValid+=1 
    if (x, y) in valid_tiles:     
        valid_actions[4] = 1 #WAIT
        nValid+=1 
    free_space = field == 0

    # 1) coins
    coin_move = torch.zeros(6)
    tile,distance,path_is_free = look_for_targets(free_space, (x,y), coins)
    if tile in valid_tiles:
        if not path_is_free:
            distance *= 2
        if tile == (x, y - 1): coin_move[2] = 1 / distance  #UP
        if tile == (x, y + 1): coin_move[3] = 1 / distance  #DOWN
        if tile == (x - 1, y): coin_move[0] = 1 / distance  #LEFT
        if tile == (x + 1, y): coin_move[1] = 1 / distance  #RIGHT
    else:
        if nValid > 0: coin_move = valid_actions / nValid

    # 2) destroy crates
    # first check for dead ends
    checkSingleCrates=False
    noGoodPositionFound = False
    destroy_crates_move = torch.zeros(6)
    if len(dead_ends) > 0:
        if (x, y) in dead_ends and (x,y) in valid_tiles: 
                if canBomb: destroy_crates_move[5] = 1 #BOMB
                else:       destroy_crates_move[4] = 1 #WAIT
        else:
            tile,distance,path_is_free = look_for_targets(free_space, (x,y), dead_ends)
            if path_is_free and (tile in valid_tiles):
                if tile == (x, y - 1): destroy_crates_move[2] = 1 / distance  #UP
                if tile == (x, y + 1): destroy_crates_move[3] = 1 / distance  #DOWN
                if tile == (x - 1, y): destroy_crates_move[0] = 1 / distance  #LEFT
                if tile == (x + 1, y): destroy_crates_move[1] = 1 / distance  #RIGHT
            else:
                checkSingleCrates = True
    else: 
        checkSingleCrates = True

    if checkSingleCrates and len(crates) > 0:
        tile,distance,path_is_free = look_for_targets(free_space, (x,y), crates)
        if distance <= 1 and (x,y) in valid_tiles:
            if canBomb: destroy_crates_move[5] = 1 #BOMB if path to dead end is not free and standing next to crate
            else:       destroy_crates_move[4] = 1 #WAIT
        else: 
            if (tile in valid_tiles) and path_is_free: #else: move to next crate if possible
                if tile == (x, y - 1): destroy_crates_move[2] = 1 / distance  #UP
                if tile == (x, y + 1): destroy_crates_move[3] = 1 / distance  #DOWN
                if tile == (x - 1, y): destroy_crates_move[0] = 1 / distance  #LEFT
                if tile == (x + 1, y): destroy_crates_move[1] = 1 / distance  #RIGHT
            else:
                noGoodPositionFound = True
        
    if noGoodPositionFound:
        if nValid > 0: coin_move = valid_actions / nValid
        
            

    # 3) others
    others_move = torch.zeros(6)
    tile,distance,path_is_free = look_for_targets(free_space, (x,y), others)
    if tile is None:
        if nValid > 0: coin_move = valid_actions / nValid
    else:
        if distance <= 1:
            if canBomb: others_move[5] = 1 #BOMB
            else: 
                if nValid > 0: coin_move = valid_actions / nValid
        else:
            if not path_is_free:
                distance *= 2
            if tile == (x, y - 1): coin_move[2] = 1 / distance  #UP
            if tile == (x, y + 1): coin_move[3] = 1 / distance  #DOWN
            if tile == (x - 1, y): coin_move[0] = 1 / distance  #LEFT
            if tile == (x + 1, y): coin_move[1] = 1 / distance  #RIGHT

    
    features = torch.cat((coin_move, destroy_crates_move, others_move)) #3*6 features
    return features.unsqueeze(0)


def look_for_targets(free_space, start, targets, logger=None):
    """Find direction of closest target that can be reached via free tiles.

    Performs a breadth-first search of the reachable free tiles until a target is encountered.
    If no target can be reached, the path that takes the agent closest to any target is chosen.

    Args:
        free_space: Boolean numpy array. True for free tiles and False for obstacles.
        start: the coordinate from which to begin the search.
        targets: list or array holding the coordinates of all target tiles.
        logger: optional logger object for debugging.
    Returns:
        coordinate of first step towards closest target or towards tile closest to any target.
    """
    if len(targets) == 0: return None,None,None

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()


    path_is_free = False
    while len(frontier) > 0:
        current = frontier.pop(0)
        # Find distance from current position to all targets, track closest
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            # Found path to a target's exact position, mission accomplished!
            best = current
            best_dist = dist_so_far[current]
            path_is_free = True
            break
        # Add unexplored free neighboring tiles to the queue in a random order
        x, y = current
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if free_space[x, y]]
        shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1
    if logger: logger.debug(f'Suitable target found at {best}')
    # Determine the first step towards the best found target tile
    current = best
    while True:
        if parent_dict[current] == start: return current, best_dist, path_is_free
        current = parent_dict[current]   