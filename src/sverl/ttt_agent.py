import numpy as np
from collections import defaultdict
from TTTenv import TTT
from tqdm import trange
import copy
    

#This doesn't learn, don't know why.

class Agent:
    """
    Usual Q learning agent, with added functionality:
        - for calculting Shapley values.
        - for environments where action space is state dependent.
    """
    
    def __init__(self, state_dim, num_actions, epsilon=0.05, gamma=0.99, alpha=0.2):
        
        self.num_actions = num_actions
        self.state_dim = state_dim
        self.actions = np.arange(self.num_actions)
        self.valid_dict = ValidDict()

        # Agent hyper-parameters
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        
        self.Q_table = defaultdict(lambda: np.zeros(self.num_actions))

    def choose_action(self, state, exp=False):
        """
        Chooses action with epsilon greedy policy.
        Info contains which actions are available in the state.
        """
        valid_actions = self.valid_dict[state.tobytes()]
        state = state.flatten()
        if np.random.rand() < self.epsilon and exp: return np.random.choice(valid_actions)
        else: 
            q_values = self.Q_table[tuple(state)][valid_actions]
            return np.random.choice(valid_actions[q_values == q_values.max()])
        
    def update(self, state, action, new_state, reward, done):
        """
        Q learning update. Only look ahead over available actions.
        """
        new_state = new_state.flatten()
        state = state.flatten()
        valid_actions = self.valid_dict[state.tobytes()]
        # Clause to stop error when valid "actions" is empty at end of episode.
        if done: q_max = 0
        else: q_max = self.Q_table[tuple(new_state)][valid_actions].max()
        
        # Usual update, for only valid "actions"
        td_error = reward + self.gamma * q_max - self.Q_table[tuple(state)][action]
        self.Q_table[tuple(state)][action] += self.alpha * td_error


# ------------------------------------------------------------------ Usual agent stuff finishes.

class PolicyDict(dict):
    """
    Special policy dictionary which sets the initial policy to random between available actions.
    """
    
    def __missing__(self, key):

        valid_actions = self.valid_dict[np.array(key).tobytes()]
        val = np.zeros(self.num_actions)
        val[valid_actions] = 1 / len(valid_actions)


        self.__setitem__(key, val)
        
        return val

class ValidDict(dict, TTT):
    
    def __missing__(self, key):
        
        val = self.valid_actions(np.frombuffer(key, dtype=np.int_))
        self.__setitem__(key, val)
        
        return val
    


def train(agent, env, num_steps):
    """
    Trains an agent for a set number of steps.
    """

    state, info = env.reset()

    for _ in trange(int(num_steps)):

        # Usual RL, choose action, execute, update
        action = agent.choose_action(state, exp = True)
        new_state, reward, terminated, truncated, info = env.step(action)
        agent.update(state, action, new_state, reward, terminated or truncated)
        state = new_state

        if terminated or truncated: state, info = env.reset()