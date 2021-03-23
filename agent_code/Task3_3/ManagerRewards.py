import events as e

def reward_from_events(self, events) -> int:

    # game_rewards = {
    #     e.COIN_COLLECTED: 500, #100,
    #     e.KILLED_OPPONENT: 500,
    #     e.MOVED_RIGHT: -1,
    #     e.MOVED_LEFT: -1,
    #     e.MOVED_UP: -1,
    #     e.MOVED_DOWN: -1,
    #     e.WAITED: -5,
    #     e.INVALID_ACTION: -5,
    #     e.BOMB_DROPPED: -1,
    #     e.KILLED_SELF: -500,
    #     e.GOT_KILLED: -500,
    #     e.CRATE_DESTROYED: 100
    # }
    game_rewards = {
        e.COIN_COLLECTED: 100,
        e.KILLED_OPPONENT: 500,
        e.MOVED_RIGHT: -1,
        e.MOVED_LEFT: -1,
        e.MOVED_UP: -1,
        e.MOVED_DOWN: -1,
        e.WAITED: -1,
        e.INVALID_ACTION: -10,
        e.BOMB_DROPPED: -1,
        e.KILLED_SELF: 0,
        e.GOT_KILLED: -700,
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
    reward_sum += crate_rewards(self, events)

    self.logger.info(f"Awarded {reward_sum} for own transition events")
    return reward_sum

# Give the crate rewards imediately after placing the crates. This makes our agent place bombs more often (better)
def crate_rewards(self, events):
    if e.BOMB_DROPPED in events:
        self.logger.info(f"reward for the {self.destroyed_crates} that are going to be destroyed -> +{self.destroyed_crates * 33}")
        return self.destroyed_crates * 10
    return 0