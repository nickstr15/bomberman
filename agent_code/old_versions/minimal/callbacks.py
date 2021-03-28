import numpy as np
from tensorflow.keras.models import load_model

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

def setup(self):
    self.model=load_model("model")
    
def act(self, game_state):
    return ACTIONS[np.argmax(self.model.predict(transformfield(game_state)))]





























def transformfield(game_state):
    dist=7
    field=-np.ones((2*dist+1,2*dist+1))
    me=game_state["self"][3]
    xmin=max(me[0]-dist,0)      #magic
    ymin=max(me[1]-dist,0)
    xmax=min(me[0]+dist+1,17)   #more CoOrDs
    ymax=min(me[1]+dist+1,17)
    fieldxmin=max(dist-me[0],0) #random maxmins
    fieldymin=max(dist-me[1],0)
    fieldxmax=min(17+dist-me[0],2*dist+1)
    fieldymax=min(17+dist-me[1],2*dist+1)
    bombs=game_state["bombs"]
    others=game_state["others"]
    newfield=np.zeros((17,17))
    coins=game_state["coins"]
    for coin in coins:     
        newfield[coin]=10
    for other in others:
        newfield[other[3]]=2
    for bomb in bombs:
        newfield[bomb[0]]=-5+bomb[1] #some calculation
    field[fieldxmin:fieldxmax,fieldymin:fieldymax]=(game_state["field"]+newfield)[xmin:xmax,ymin:ymax]      #MoRe InDeXaTiOn
    return field.reshape(1,-1)
