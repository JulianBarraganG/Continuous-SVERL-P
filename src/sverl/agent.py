import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import trange
import cma

import numpy as np

# Model definition
class Policy_cartpole(nn.Module):
    def __init__(self, state_space_dimension, action_space_dimension, num_neurons=5, bias = False):
        super(Policy_cartpole, self).__init__()
        self.state_space_dimension = state_space_dimension
        self.fc = nn.Linear(state_space_dimension, num_neurons, bias=bias)
        self.fc1 = nn.Linear(num_neurons, action_space_dimension, bias=bias)

    def forward(self, x):
        x = torch.Tensor( x.reshape((1, self.state_space_dimension)) )
        hidden = torch.tanh(self.fc(x))
        output = self.fc1(hidden)
        return int(output>0)
    
#This just comes from the CMA assignment
def fitness_cart_pole(x, nn, env):
    '''
    Returns negative accumulated reward for single pole, fully environment.

    Parameters:
        x: Parameter vector encoding the weights.
        nn: Parameterized model.
        env: Environment ('CartPole-v?').
    '''
    torch.nn.utils.vector_to_parameters(torch.Tensor(x), nn.parameters())  # Set the policy parameters
    state_space_dimension = env.observation_space.shape[0]  # State space dimension
    state = env.reset()[0]  # Forget about previous episode
    
          
    R = 0  # Accumulated reward
    while True:
        a = nn(state)
        
        state, reward, terminated, truncated, _ = env.step(a)  # Simulate pole
        R += reward  # Accumulate 
        if truncated:
            return -1000  # Episode ended, final goal reached, we consider minimization
        if terminated:
            return -R  # Episode ended, we consider minimization
    return -R  # Never reached  



#Again, just comes from the assignment
def train_cartpole_agent(policy_net , env):     
    d = sum(param.numel() for param in policy_net.parameters())
    initial_weights = np.random.normal(0, 0.01, d)  # Random parameters for initial policy, d denotes the number of weights
    initial_sigma = .01 # Initial global step-size sigma
    # Do the optimization
    res = cma.fmin(fitness_cart_pole,  # Objective function
                initial_weights,  # Initial search point
                initial_sigma,  # Initial global step-size sigma
                args=([policy_net, env]),  # Arguments passed to the fitness function
                options={'ftarget': -999.9, 'tolflatfitness':1000, 'eval_final_mean':False})
    env.close()
  
    # Set the policy parameters to the final solution
    torch.nn.utils.vector_to_parameters(torch.Tensor(res[0]), policy_net.parameters())      

    return policy_net  # Return the policy network





