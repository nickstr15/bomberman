import events as e

from .ManagerFeatures import closest_coin

def reward_from_events(self, events) -> int:

    game_rewards = {
        e.COIN_COLLECTED: 100,
        e.KILLED_OPPONENT: 5,
        e.MOVED_RIGHT: -0.01,
        e.MOVED_LEFT: -0.01,
        e.MOVED_UP: -0.01,
        e.MOVED_DOWN: -0.01,
        e.WAITED: -0.5,
        e.INVALID_ACTION: -0.5,
        e.BOMB_DROPPED: -0.01,
        e.KILLED_SELF: -20,
        e.GOT_KILLED: -10,
        e.CRATE_DESTROYED: 0.5
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
    reward_sum += moved_closer_to_next_coin(old_game_state, action, events)

    self.logger.info(f"Awarded {reward_sum} for own transition events")
    return reward_sum

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
