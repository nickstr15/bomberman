import numpy as np

class Model():

    def __init__(self, feature_count, n, gamma, alpha,state_to_features, parameter_vectors):
        '''
        feature_count: number of features

        n: number of steps in temporal difference Q-learning
        gamma: discounting factor
        alpha: learning rate

        date_to_features: function that receives current state as input
        parameter_vectors: start vectors for training
        '''

        self.feature_count = feature_count
        self.state_to_features = state_to_features
        self.parameter_vectors = parameter_vectors

        self.n = n
        self.gamma = gamma
        self.alpha = alpha

        self.discounts = (np.ones((n+1))*gamma)**np.linspace(0,n,n+1) #discounting factors 

        self.batches = [[] for _ in range(6)]  #six batches for the six moves
        self.rewards = [[]]
        self.states = [[]]

        self.TAU = 0
        self.T = 400 #maximum 400 time steps per episode

    def add_step(self, tau, t, state, action, reward):
        '''
        tau: episode
        t: time step in episode
        action: action that is executed
        reward: reward that is earned after the action
        '''
        if tau > self.TAU: #new episode
            self.TAU = TAU
            self.rewards.append([])
            self.states.append([])

        self.batches[action].append((tau, t))
        self.rewards[tau].append(reward)
        self.states[tau].append(self.state_to_features(state))

    def update_parameter_vectors(self):
        #calculate The Ys
        Y = np.empty((self.TAU, self.T))
        Y.fill(np.nan)
        for tau in range(self.TAU):
            for t in range(self.t):
                if t+self.n < self.T:
                    rewards = np.array(self.rewards[tau][t:t+self.n]) #rewards for next n steps
                    Y[tau][t] = np.nansum(rewards*self.discounts[0:-1]) 
                    Y[tau][t] += np.max(np.dot(self.states[tau][t],self.parameter_vectors))*self.discounts[-1]
                else:
                    rewards = np.array(self.rewards[tau][t:]) #rewards till end of the game
                    discounts = np.array(self.discounts[0:len(self.rewards[tau][t:])]) #discounts for these rewards
                    Y[tau][t] = np.nansum(rewards*discounts)

        #update the parameter vectors
        for i,beta in enumerate(self.parameter_vectors):
            batch = self.batches[i]
            sum = np.zeros(len(beta))
            for tau, t in batch:
                sum += self.states[tau][t] * (Y[tau][t] - np.dot(self.states[tau][t], beta))
            self.parameter_vectors[i] += (self.alpha / len(batch)) * sum

        #reset for next training cicle
        self.batches = [[] for _ in range(6)]  #six batches for the six moves
        self.rewards = [[]]
        self.states = [[]]

        self.TAU = 0

    def predict_action(self, state):
        #return softmax of the actions for given state
        Q = np.dot(self.state_to_features(state), self.parameter_vectors) 
        return np.exp(Q)/sum(np.exp(Q))

        
