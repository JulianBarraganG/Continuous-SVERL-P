import numpy as np
from collections import defaultdict
from ttt_env import TTT, ScoreDict
from tqdm import trange
import copy
    

#This doesn't learn, don't know why.

class Agent:    
    def __init__(self):
        self.score_dict = ScoreDict()
        self.score_dict.score_dict = self.score_dict
    
    def train(self, env): 
        starting_states = [np.zeros(9)]

        for i in range(9): 
            pos = np.zeros(9)
            pos[i] = 1
            starting_states.append(pos)
            pos = np.zeros(9)
            pos[i] = 2
            starting_states.append(pos)
        for state in starting_states: 
            _, _ = env.reset(start_state = state)
            env.minmax_player()

        self.score_dict = env.get_optimal_policy_dict()         

    def choose_action(self, state): 
        if(state.shape == (9,)): 
            state = state.reshape(3,3)
        print(state)
        _, best_moves = self.score_dict[tuple([state.tobytes(), 1])]
        return np.random.choice(best_moves)
