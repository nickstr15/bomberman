import events as e

from .ManagerFeatures import closest_coin

def reward_from_events(self, events) -> int:

    game_rewards = {
        e.COIN_COLLECTED: 100,
        e.KILLED_OPPONENT: 100,
        e.MOVED_RIGHT: -0.01,
        e.MOVED_LEFT: -0.01,
        e.MOVED_UP: -0.01,
        e.MOVED_DOWN: -0.01,
        e.WAITED: -0.5,
        e.INVALID_ACTION: -0.5,
        e.BOMB_DROPPED: -0.01,
        e.KILLED_SELF: -300,
        e.GOT_KILLED: -300
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

def rewards_from_own_events(self, old_game_state, action, new_game_state):
    reward_sum = 0

    # check if agent moved closer to next coin
    if len(old_game_state['coins']) > 0:
        if moved_closer_to_next_coin(old_game_state, action):
            reward_sum += 0.05
        else:
             reward_sum -= 0.06

    self.logger.info(f"Awarded {reward_sum} for own transition events")
    return reward_sum

def moved_closer_to_next_coin(old_game_state, action):
    agent_x, agent_y = agent_x, agent_y = old_game_state['self'][3]
    coin = closest_coin(agent_x, agent_y, old_game_state['coins'])
    if   (coin[0] == 1) and (action == 'RIGHT'): return True
    elif (coin[1] == 1) and (action == 'LEFT'):  return True
    elif (coin[2] == 1) and (action == 'DOWN'):  return True
    elif (coin[3] == 1) and (action == 'UP'):    return True
    else: return False
