import pickle
import random
from collections import namedtuple, deque
from typing import List
import torch.optim as optim 
import events as e
from .callbacks import state_to_features
import torch
from .NNModel import state_to_features, reward_from_events, train, save_parameters

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
#TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
#RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
#PLACEHOLDER_EVENT = "PLACEHOLDER"

ACTIONS_IDX = {'LEFT':0, 'RIGHT':1, 'UP':2, 'DOWN':3, 'WAIT':4, 'BOMB':5}
#ACTIONS_IDX = {'LEFT':0, 'RIGHT':1, 'UP':2, 'DOWN':3, 'WAIT':4}

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    #self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.iteration_counter = 0
    self.train_counter = 0

    self.experience_buffer = [[]]
    self.tau = 0
    self.first = True
    self.network.optimizer = self.network.optimizer(self.network.parameters(), lr=self.network.alpha)



def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """

    #self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # previous_state = state_to_features(old_game_state)
    # if previous_state is not None:
    #     action_idx = ACTIONS_IDX[self_action]
    #     reward = reward_from_events(self, events)

    #     self.experience_buffer[self.tau].append((previous_state, action_idx, reward)) #add game experience
    #     number_of_elements_in_buffer = sum([len(episode) for episode in self.experience_buffer])
    #     if number_of_elements_in_buffer > self.network.buffer_size:          #delte first experince if buffer is full
    #         if len(self.experience_buffer[0]) == 0:
    #             self.experience_buffer.pop(0)
    #             self.tau -= 1
    #         self.experience_buffer[0].pop(0)
    old_state = state_to_features(old_game_state)
    if old_state is not None:
        new_state = state_to_features(new_game_state)
        reward = reward_from_events(self, events)

        action_idx = ACTIONS_IDX[self_action]
        action = torch.zeros(6)
        action[action_idx] = 1

        
        self.experience_buffer[self.tau].append((old_state, action, reward, new_state))
        number_of_elements_in_buffer = sum([len(episode) for episode in self.experience_buffer])
        if number_of_elements_in_buffer > self.network.buffer_size:
            if len(self.experience_buffer[0]) == 0:
                self.experience_buffer.pop(0)
                self.tau -= 1
            self.experience_buffer[0].pop(0)
        
        train(self.network, self.experience_buffer)
        self.train_counter +=1
        if self.train_counter % 5000 == 0:
            print(f"{self.train_counter} parameter updates")

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.iteration_counter +=1

    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    #self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    if self.iteration_counter % 100 == 0:
        save_parameters(self.network, self.iteration_counter)

    self.tau += 1
    self.experience_buffer.append([]) #new buffer for next episode

    self.counter +=1


    
    

        



