import numpy as np
from collections import defaultdict
from ttt_env import TTT, ScoreDict, WonDict, ValidDict
from tqdm import trange
import copy
    

#This doesn't learn, don't know why.

class Agent:    
    def __init__(self):
        self.action_dict = {0: (0, 0),
                            1: (0, 1),
                            2: (0, 2),
                            3: (1, 0),
                            4: (1, 1),
                            5: (1, 2),
                            6: (2, 0),
                            7: (2, 1),
                            8: (2, 2)}
        self.won_dict = WonDict()

        # Save previously calculated valid actions for boards, also speed.
        self.valid_dict = ValidDict()

        # Save previous scores for minmax, speed.
        self.score_dict = ScoreDict()
        self.score_dict.won_dict = self.won_dict
        self.score_dict.valid_dict = self.valid_dict
        self.score_dict.action_dict = self.action_dict
        self.score_dict.score_dict = self.score_dict
    
    def score(self, state, player):
        """
        Given the game state and whose turn it is returns a tuple (estimated game score, best move to play)
        """

        state_byte = state.tobytes()

        done, winner = self.won_dict[state_byte]

        if not done: end_score = None
        else: end_score = (winner + 1) % 3 - 1

        if end_score is not None: return end_score, None
        else:
            all_moves = self.valid_dict[state_byte]
                        
            scores = np.empty(len(all_moves))
            
            n_player = player % 2 + 1

            for i, action in enumerate(all_moves):

                new_state = state.copy()
                new_state[self.action_dict[action]] = player

                current_score, _ = self.score_dict[tuple([new_state.tobytes(), n_player])]

                scores[i] = current_score

            if player == 1: best_score = max(scores)
            elif player == 2: best_score = min(scores)

            best_moves = all_moves[scores == best_score]
                        
            return best_score, best_moves      

    def choose_action(self, state): 
        if(state.shape == (9,)): 
            state = state.reshape(3,3)
        _, best_moves = self.score_dict[tuple([state.tobytes(), 1])]
        if(best_moves is None): 
            #If the agent thinks there are no legal moves, but there actually are (The random sampler
            #has sampled 1 where it should be 0 for example) - what do we do in this case?
        return np.random.choice(best_moves)
