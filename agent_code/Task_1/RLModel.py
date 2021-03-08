import numpy as np
import pickle
class Model():

    def __init__(self, feature_count, n, gamma, alpha, parameter_vectors):
        '''
        feature_count: number of features

        n: number of steps in temporal difference Q-learning
        gamma: discounting factor
        alpha: learning rate

        parameter_vectors: start vectors for training
        '''

        self.feature_count = feature_count
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
            self.TAU = tau
            self.rewards.append([])
            self.states.append([])

        self.batches[action].append((tau, t))
        self.rewards[tau].append(reward)
        self.states[tau].append(state)

    def update_parameter_vectors(self):
        #print(self.parameter_vectors)
        #calculate The Ys
        Y = np.empty((self.TAU+1, self.T))
        Y.fill(np.nan)
        for tau in range(self.TAU+1):
            steps = len(self.states[tau])
            for t in range(steps-self.n):
                rewards = np.array(self.rewards[tau][t:t+self.n]) #rewards for next n steps
                Y[tau][t] = np.nansum(rewards*self.discounts[0:-1]) 
                Y[tau][t] += np.max(np.dot(self.states[tau][t+self.n],self.parameter_vectors.T))*self.discounts[-1]
            for t in range(steps-self.n,steps):
                rewards = np.array(self.rewards[tau][t:]) #rewards till end of the game
                discounts = np.array(self.discounts[0:len(self.rewards[tau][t:])]) #discounts for these rewards
                Y[tau][t] = np.nansum(rewards*discounts)

        #update the parameter vectors
        for i,beta in enumerate(self.parameter_vectors):
            batch = self.batches[i]
            sum_ = np.zeros(len(beta))
            #print(beta)
            for tau, t in batch:
                if np.isnan(Y[tau][t]):
                    print(tau, t)
                sum_ += self.states[tau][t] * (Y[tau][t] - np.dot(self.states[tau][t], beta))
            #print(sum_)
            self.parameter_vectors[i] += (self.alpha / len(batch)) * sum_

        #reset for next training cicle
        self.batches = [[] for _ in range(6)]  #six batches for the six moves
        self.rewards = [[]]
        self.states = [[]]

        self.TAU = 0

        #print("***\nUpdated:")
        #print(self.parameter_vectors)

        # Store the model
        with open("my-saved-model.pt", "wb") as file:
            pickle.dump(self.parameter_vectors, file)

    def predict_action(self, state):
        #return softmax of the actions for given state
        Q = np.dot(state, self.parameter_vectors.T) 
        #print("***")
        #print(Q)
        #print("softmax")
        #print(np.exp(Q)/sum(np.exp(Q)))
        Q /= np.max(np.abs(Q))
        return Q, np.exp(Q)/sum(np.exp(Q)), np.argmax(Q) #Q, softmax, max action index

        
