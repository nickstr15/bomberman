import numpy as np

def act_rulebased(self):
    '''
    returns action based on manual linear feature combination
    
    :param self: This object is passed to all callbacks and you can set arbitrary values.
    '''

    ACTIONS = ['RIGHT', 'LEFT', 'DOWN','UP',  'WAIT', 'BOMB']
    
    Q = np.dot(self.action_array, self.features)
    action = ACTIONS[np.argmax(Q)]
    print()
    print(np.array([ACTIONS, Q]).T)
    print(f"--> {action}")
    return action

def initialize_rule_based(self):
    '''
    create the matrix for linear rule based decision
    The entries are based on our rewards and were found by try and error
    
    :param self: This object is passed to all callbacks and you can set arbitrary values.
    '''
    self.action_array = np.zeros((6,23))
    # coins
    self.action_array[0][0] = 100
    self.action_array[1][1] = 100
    self.action_array[2][2] = 100
    self.action_array[3][3] = 100

    # crates
    self.action_array[0][4] = 33
    self.action_array[1][5] = 33
    self.action_array[2][6] = 33
    self.action_array[3][7] = 33

    # bomb here
    self.action_array[5][8] = 36

    # explosion here
    self.action_array[0][9] = 10
    self.action_array[1][9] = 10
    self.action_array[2][9] = 10
    self.action_array[3][9] = 10
    self.action_array[4][9] = -10
    self.action_array[5][9] = -10

    # run away
    self.action_array[0][10] = 300
    self.action_array[1][11] = 300
    self.action_array[2][12] = 300
    self.action_array[3][13] = 300
    self.action_array[4][10] = 300
    self.action_array[4][11] = 300
    self.action_array[4][12] = 300
    self.action_array[4][13] = 300
    self.action_array[5][10] = 300
    self.action_array[5][11] = 300
    self.action_array[5][12] = 300
    self.action_array[5][13] = 300

    # not run in explosion
    self.action_array[0][14] = 400
    self.action_array[1][15] = 400
    self.action_array[2][16] = 400
    self.action_array[3][17] = 400

    # walk to opponent
    self.action_array[0][18] = 75
    self.action_array[1][19] = 75
    self.action_array[2][20] = 75
    self.action_array[3][21] = 75