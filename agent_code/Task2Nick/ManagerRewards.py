import events as e
from .ManagerFeatures import closest_coin

def reward_from_events(self, events) -> int:

    game_rewards = {
        e.COIN_COLLECTED: 500, #100,
        e.KILLED_OPPONENT: 500,
        e.MOVED_RIGHT: -1,
        e.MOVED_LEFT: -1,
        e.MOVED_UP: -1,
        e.MOVED_DOWN: -1,
        e.WAITED: -5,
        e.INVALID_ACTION: -5,
        e.BOMB_DROPPED: -1,
        e.KILLED_SELF: -500,
        e.GOT_KILLED: -500,
        e.CRATE_DESTROYED: 100
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
    reward_sum += loop_killer(self, new_game_state)

    self.logger.info(f"Awarded {reward_sum} for own transition events")
    return reward_sum


###################
## sub functions ##
###################

def moved_closer_to_next_coin(old_game_state, action, events):
    if e.INVALID_ACTION in events:
        return 0
        
    good, bad = 0.05, -0.06

    agent_x, agent_y = agent_x, agent_y = old_game_state['self'][3]
    coin = closest_coin(agent_x, agent_y, old_game_state['coins'])
    if   (coin[0] == 1) and (action == 'RIGHT'): return good
    elif (coin[1] == 1) and (action == 'LEFT'):  return good
    elif (coin[2] == 1) and (action == 'DOWN'):  return good
    elif (coin[3] == 1) and (action == 'UP'):    return good
    else: return bad

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
