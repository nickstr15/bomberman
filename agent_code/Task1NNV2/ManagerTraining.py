
import random
import numpy as np
import torch 
    

def generate_eps_greedy_policy(network):
    return np.linspace(network.epsilon_begin, network.epsilon_end, network.training_episodes)

def update_network(network, experience_buffer):
    '''
    network: the network that gets updated
    experience_buffer: the collected experiences, list of game_episodes
    '''

    #randomly choose batch out of the experience buffer
    number_of_elements_in_buffer = len(experience_buffer)
    batch_size = min(number_of_elements_in_buffer, network.batch_size)

    random_i = [ random.randrange(number_of_elements_in_buffer) for _ in range(batch_size)]


    #compute for each experience in the batch 
    # - the Ys using n-step TD Q-learning
    # - the current guess for the Q function
    sub_batch = []
    Y = []
    for i in random_i:
        random_experience = experience_buffer[i]
        sub_batch.append(random_experience)
    
    for b in sub_batch:
        old_state = b[0]
        action = b[1]
        reward = b[2]
        new_state = b[3]

        y = reward
        if new_state is not None:
            y += network.gamma * torch.max(network(new_state))

        Y.append(y)

    Y = torch.tensor(Y)

    #Qs
    states = torch.cat(tuple(b[0] for b in sub_batch))  #put all states of the sub_batch in one batch
    q_values = network(states)
    actions = torch.cat([b[1].unsqueeze(0) for b in sub_batch])
    Q = torch.sum(q_values*actions, dim=1)
    
    loss = network.loss_function(Q, Y)
    network.optimizer.zero_grad()
    loss.backward()
    network.optimizer.step()

def save_parameters(network, string):
    torch.save(network.state_dict(), f"network_parameters/{string}.pt")