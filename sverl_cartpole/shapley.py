import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from NeuralConditioner import predict_missing_features
from math import factorial
from utils import get_all_subsets

import cma

import numpy as np

#Basically the local sverl. Uses the neural conditioner to predict missing features in the first step, and then has full observability afterwards. 
#Very uninteresting to be honest, since the cart pole can only go left 0, or right 1. And even though the model gives different values, the decision
#Will usually still be the same, just with more or less certainty. And even if the missing features leads to a bad decision 
#In the initital step, it can be saved, so it doesn't really matter much. 
#I haven't used this function much
def local_sverl_value_function(policy, seed, NC, mask, env):
    ''''
    'Evaluate the policy from a given state, using the believed state to make the initial decision'
    '''
    R = 0
    true_state = env.reset(seed=seed)[0]  # Forget about previous episode
    state_space_dimension = env.observation_space.shape[0]

    pred = predict_missing_features(NC, true_state, mask)


    believed_initial_state = np.copy(true_state)

    for i in range(state_space_dimension):
        if mask[i] == 0:
            believed_initial_state[i] = pred[i]
    
    


    print("Believed state: ", believed_initial_state)
    print("True state: ", true_state)

    print("Action if evaluated on true state: ", policy(true_state))
    print("Action if evaluated on believed state: ", policy(believed_initial_state))

    a = policy(believed_initial_state)
    
    state, reward, terminated, truncated, _ = env.step(a)  # Simulate pole
    R +=reward
    state_tensor = torch.Tensor( state.reshape((1, state_space_dimension)) ) 
    

    while True:
        a = policy(state_tensor)
        
        state, reward, terminated, truncated, _ = env.step(a)  # Simulate pole
        R+=reward
        state_tensor = torch.Tensor( state.reshape((1, state_space_dimension)) )
        if(terminated or truncated): 
            break
        
    env.close()
    return R  # Return the cumulated reward


#Now this is juicy. Gives a global evaluation of the policy, but with missing features.
#In every step, it uses the neural conditioner to predict the missing features, and then uses the policy to decide what to do.

def global_sverl_value_function(policy, seed, NC, mask, env):
    ''''
    'Evaluate the policy from a given state, using the believed state to make the initial decisino'
    '''
    R = 0
    true_state = env.reset(seed=seed)[0]  # Forget about previous episode
    state_space_dimension = env.observation_space.shape[0]  # State space dimension
    
    pred = predict_missing_features(NC, true_state, mask)

  
    believed_state = np.copy(true_state)
  

    for i in range(state_space_dimension):
        if mask[i] == 0:
            
            believed_state[i] = pred[i]

    while True:       

        a = policy(believed_state)
        
        state, reward, terminated, truncated, _ = env.step(a)  # Simulate pole
        R+=reward
        pred = predict_missing_features(NC, state, mask)
        believed_state = np.copy(state)
        for i in range(state_space_dimension):
            if mask[i] == 0:
                believed_state[i] = pred[i]

        if(terminated or truncated): 
            break
        
    env.close()
    return R #Return the cumulated reward


#Takes a feature i, and a mask C. Gives the marginal gain of adding feature i to the mask C, via an evaluation function
def marginal_gain(policy, neural_conditioner,eval_function , features, C, seed, env): 
    C_i = np.copy(C)
    for i in features:
        C_i[i] = 1


    """ if(np.sum(C) == 0): 
        # v(Ø) = 0 for game theory, and for ML v(Ø) = expected prediction of model.
        # We have yet to determine what v(Ø) is for SVERL, and for the general RL case.
        V_C = 0
    else: 
        V_C = eval_with_missing_features(policy, seed, NC, C) """

    V_C = eval_function(policy, seed, neural_conditioner, C, env)

    V_C_i= eval_function(policy, seed, neural_conditioner, C_i, env)

    return V_C_i - V_C


#Calculates Shapley values for a feature, using the marginal gain function and the get_all_subsets function.
def shapley_value(policy, neural_conditioner, eval_function,  features, seed, env):
    state_space = env.observation_space.shape[0]
    list_of_C = get_all_subsets(features, state_space)
    sum = 0

    state_space = env.observation_space.shape[0]

    for C in list_of_C:
        sum += marginal_gain(policy, neural_conditioner, eval_function, features, C, seed, env)* ((factorial(np.sum(C))*factorial(state_space - np.sum(C) - 1)) / factorial(state_space))
    return sum
