

import torch
import numpy as np
from collections import deque
import bisect
import heapq
import random

import time

ROWS = 11
COLS = 11

def state_to_features(game_state: dict) -> torch.tensor:
    '''
    converts the game_state dict in a RGB image 
    that is returned as a tensor
    '''
    if game_state is None:
        return None

    agent = torch.zeros(ROWS,COLS)
    xy = game_state['self'][3]
    hasBomb = game_state['self'][2]
    agent[xy] = 2 if hasBomb else 1

    walls = torch.zeros(ROWS,COLS)
    walls_mask = torch.tensor(game_state['field'] == -1)
    walls[walls_mask] = -1

    crates = torch.zeros(ROWS,COLS)
    crates_mask = game_state['field'] == 1
    walls[walls_mask] = 1

    coins = torch.zeros(ROWS,COLS)
    for xy in game_state['coins']:
        coins[xy] = 1

    death = torch.zeros(ROWS,COLS)
    for (bx,by),t in game_state['bombs']:
        for x in range(bx-3,bx+4):
            if bx+x < COLS-1 and bx+x > 0:
                death[bx+x,by] = -1
        for y in range(by-3,by+4):
            if by+y < ROWS-1 and by+y > 0:
                death[bx,by+y] = -1

    others = torch.zeros(ROWS,COLS)
    for _,_,hasBomb,xy in game_state['others']:
        others[xy] = 2 if hasBomb else 1


    features = torch.stack((agent,walls,crates,coins,death,others))
    return features.unsqueeze(0)