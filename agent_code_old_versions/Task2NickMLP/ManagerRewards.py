import events as e

def reward_from_events(self, events) -> int:

    game_rewards = {
        e.COIN_COLLECTED: 500,
        e.KILLED_OPPONENT: 100,
        e.MOVED_RIGHT: -1,
        e.MOVED_LEFT: -1,
        e.MOVED_UP: -1,
        e.MOVED_DOWN: -1,
        e.WAITED: -1,
        e.INVALID_ACTION: -2,
        e.BOMB_DROPPED: -1,
        e.KILLED_SELF: -100,
        e.GOT_KILLED: -300
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

def rewards_from_own_events(self, old_game_state, action, new_game_state, events):
    reward_sum = 0

    # check if agent moved closer to next coin
    # reward_sum += moved_closer_to_next_coin(old_game_state, action, events)
    #reward_sum += loop_killer(self, new_game_state)
    reward_sum += bomb_placed(old_game_state, events)
    reward_sum += reached_dead_end(self, new_game_state)

    #self.logger.info(f"Awarded {reward_sum} for own transition events")
    return reward_sum


###################
## sub functions ##
###################

def reached_dead_end(self, new_game_state):
    if new_game_state is None:
        return 0
    field = new_game_state['field']
    xy = new_game_state['self'][3]
    x,y = xy
    reward = 0
    if [field[x + 1, y], field[x - 1, y], field[x, y + 1], field[x, y - 1]].count(0) == 1:
        if xy not in self.dead_ends:
            reward += 100
            self.dead_ends.add(xy)
    return reward

def bomb_placed(old_game_state, events):
    agent_x, agent_y = old_game_state['self'][3]
    field = old_game_state['field']
    reward = 0
    if e.BOMB_DROPPED in events:
        for dx,dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            if field[agent_x+dx, agent_y+dy] == 1:
                reward += 33
    return reward




