from math import factorial
import numpy as np

from .shapley_utils import get_all_group_subsets, get_r

#Basically the local sverl. Uses the neural conditioner to predict missing features in the first step, and then has full observability afterwards. 
#Very uninteresting to be honest, since the cart pole can only go left 0, or right 1. And even though the model gives different values, the decision
#Will usually still be the same, just with more or less certainty. And even if the missing features leads to a bad decision 
#In the initital step, it can be saved, so it doesn't really matter much. 
#I haven't used this function much
#ms_ft_pred_fnc is a missing features prediction function (NC, RandomSampler, etc.)
def local_sverl_value_function(policy, initial_state, ms_ft_pred_fnc, mask, env):
    """
    Evaluate the policy from a given state, using the believed state to make the initial decision
    Parameters
    ----------
    policy : function
    initial_state : numpy.ndarray
    ms_ft_pred_fnc : function
    mask : np.ndarray
    env : gym.Env
    Returns
    -------
    R : float
        The cumulated reward
    """
    R = 0
    
    print(initial_state)
    believed_initial_state = ms_ft_pred_fnc(initial_state, mask)
    print(believed_initial_state)

    a = policy(believed_initial_state)
    
    state, reward, terminated, truncated, _ = env.step(a)  # Simulate pole
    R +=reward
    

    while True:
        a = policy(state)
        
        state, reward, terminated, truncated, _ = env.step(a)  # Simulate pole
        R+=reward
        if(terminated or truncated): 
            break
        
    env.close()
    return R  # Return the cumulated reward


#Now this is juicy. Gives a global evaluation of the policy, but with missing features.
#In every step, it uses the neural conditioner to predict the missing features, and then uses the policy to decide what to do.
#ms_ft_pred_fnc is a missing features prediction function (NC, RandomSampler, etc.)
def global_sverl_value_function(policy, seed, ms_ft_pred_fnc, mask, env):
    """
    Evaluate the policy with missing features, using the believed state to make the decision.
    Parameters
    ----------
    policy : callable
    seed : int
    ms_ft_pred_fnc : callable
    mask : np.ndarray
    env : gym.Env
    Returns
    -------
    R : float
        The cumulated reward
    """
    R = 0
    true_state = env.reset(seed=seed)[0]  # Forget about previous episode
    
    believed_state = ms_ft_pred_fnc(true_state.flatten(), mask)  

    while True:     
        a = policy(believed_state)
        
        state, reward, terminated, truncated, _ = env.step(a)  # Simulate pole
        R+=reward
        believed_state = ms_ft_pred_fnc(state.flatten(), mask)
       
        if(terminated or truncated): 
            break
        
    env.close()
    return R

#Takes a feature i, and a mask C. Gives the marginal gain of adding feature i to the mask C, via an evaluation function
def marginal_gain(policy, ms_ft_prd_fnc,eval_function , features, C, seed, env): 
    """
    Calculates the marginal gain of adding feature i to the mask C, using the eval_function.
    """
    C_i = np.copy(C)
    for i in features:
        C_i[i] = 1

    V_C = eval_function(policy, seed, ms_ft_prd_fnc, C, env)

    V_C_i= eval_function(policy, seed, ms_ft_prd_fnc, C_i, env)

    return V_C_i - V_C

def shapley_value(policy, ms_ft_pred_fnc, eval_function,  G, masked_group, seed, env):
    """
    Calculate the Shapley value for a feature using the marginal gain function and the get_all_subsets function.
    """
    # Cardinality integers
    num_groups = len(G) # |F| i.e. number of groups 
    num_groups_per_C = get_r(num_groups, masked_group)  # All possible |C| for all permutations excluding the masked group

    list_of_C = get_all_group_subsets(G, masked_group, r=num_groups_per_C)
    sum = 0

    for c, C in enumerate(list_of_C):
        enum = (factorial(np.sum(num_groups_per_C[c]))) * factorial(num_groups - np.sum(num_groups_per_C[c]) - 1)
        denom = factorial(num_groups)
        normalizing_constant = enum / denom
        marginal_gain_i = marginal_gain(policy, ms_ft_pred_fnc, eval_function, G[masked_group], C, seed, env)
        sum += normalizing_constant * marginal_gain_i
    return sum

