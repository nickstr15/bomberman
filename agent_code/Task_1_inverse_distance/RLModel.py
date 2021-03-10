import numpy as np
import pickle
class Model():

    def __init__(self, feature_count, gamma, alpha, parameter_vectors):
        '''
        feature_count: number of features

        n: number of steps in temporal difference Q-learning
        gamma: discounting factor
        alpha: learning rate

        parameter_vectors: start vectors for training
        '''
        self.feature_count = feature_count
        self.parameter_vectors = parameter_vectors

        self.gamma = gamma
        self.alpha = alpha

        self.experience_buffer = [] # save the TS
        self.max_buffer = 400 * 5
        self.size_buffer = 0

    def add_step(self,new_game, old_features, action, reward, new_features):
        '''
        tau: episode
        t: time step in episode
        action: action that is executed
        reward: reward that is earned after the action
        '''
        if new_game:
            self.experience_buffer.append([])

        self.experience_buffer[-1].append([old_features, action, reward, new_features])
        self.size_buffer += 1
        if self.size_buffer > self.max_buffer:
            if not len(self.experience_buffer[0]):
                self.experience_buffer.pop(0)
            self.experience_buffer[0].pop(0)

        return 0


    def update_parameter_vectors(self):
        
        Y = [[] for _ in range(6)]
        Q = [[] for _ in range(6)]
        features = [[] for _ in range(6)]
        for game in self.experience_buffer:
            for step in game:
                y = step[2] # actual reward
                new_features = step[3]
                old_features = step[0]
                action_index = step[1]
                if new_features is not None:
                    y += self.gamma * np.max(np.dot(new_features, self.parameter_vectors.T))
                Y[action_index].append(y)
                Q[action_index].append(np.dot(old_features, self.parameter_vectors.T)[action_index])
                features[action_index].append(old_features)
        action_index = None

        for action in range(6):
            batch_size = len(Y[action])
            if batch_size != 0:
                Y_a = np.array(Y[action])
                Q_a = np.array(Q[action])
                features_a = np.array(features[action])
                self.parameter_vectors[action] -= self.alpha / batch_size * np.sum(features_a.T*(Y_a - Q_a), axis=1)
        

        return 0


    def predict_action(self, state):
        #return softmax of the actions for given state
        Q = np.dot(state, self.parameter_vectors.T)
        # Q /= np.max(np.abs(Q))
        return Q, np.argmax(Q) #Q, softmax, max action index

        
