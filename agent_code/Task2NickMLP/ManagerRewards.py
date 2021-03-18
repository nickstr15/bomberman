import events as e

def reward_from_events(self, events) -> int:

    game_rewards = {
        e.COIN_COLLECTED: 100, #100,
        e.KILLED_OPPONENT: 100,
        e.MOVED_RIGHT: -1,
        e.MOVED_LEFT: -1,
        e.MOVED_UP: -1,
        e.MOVED_DOWN: -1,
        e.WAITED: -1,
        e.INVALID_ACTION: -2,
        e.BOMB_DROPPED: -1,
        e.KILLED_SELF: -100,
        e.GOT_KILLED: -300,
        e.CRATE_DESTROYED: 30
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

def rewards_from_own_events(self, old_game_state, action, new_game_state, events):
    reward_sum = 0
    return reward_sum

    # check if agent moved closer to next coin
    # reward_sum += moved_closer_to_next_coin(old_game_state, action, events)
    #reward_sum += loop_killer(self, new_game_state)
    #reward_sum += bomb_placed(old_game_state, events)

    #self.logger.info(f"Awarded {reward_sum} for own transition events")
    #return reward_sum


###################
## sub functions ##
###################


def loop_killer(self, new_game_state):
    if new_game_state is None:
        return 0
    loop = False
    if self.pos_saver.count(new_game_state["self"][3]) > 3:
        loop = True
    self.pos_saver.append(new_game_state["self"][3])
    if len(self.pos_saver) > 10:
        self.pos_saver.pop(0)
    if loop:
        return -0.5
    else: return 0

def bomb_placed(old_game_state, events):
    agent_x, agent_y = old_game_state['self'][3]
    field = old_game_state['field']
    if e.BOMB_DROPPED in events:
        if isdeadend(field, agent_x, agent_y, []):
            return 500
        reward = 0
        for x,y in [(1,0), (-1,0), (0,1), (0,-1)]:
            if field[agent_x+x, agent_y+y] == 1:
                reward += 250
        return reward
    return 0




