from gymnasium import Env
from math import factorial, log2
import numpy as np

from joblib import Parallel, delayed
from os.path import exists
from os import makedirs
import pickle
import tqdm as tqdm

from .group_utils import get_all_subsets
from .imputation_utils import evaluate_policy

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


def marginal_gain(C, i, characteristic_dict):

    """
    Calculate the marginal gain of adding feature i to the coalition C.
    
    Parameters
    ----------
    C : np.ndarray
        Coalition mask.
    C_i : np.ndarray
        Coalition mask with feature i added.
    characteristic_dict : dict
        Dictionary containing characteristic values for each coalition.

    Returns
    -------
    float
        Marginal gain of adding feature i to coalition C.
    """
    C_i = np.copy(C)
    C_i[i] = 1  # Add feature i to the coalition
    V_C = characteristic_dict[C.tobytes()]
    V_C_i = characteristic_dict[C_i.tobytes()]
    
    return V_C_i - V_C 

#Calculates Shapley values for a feature, using the marginal gain function and the get_all_subsets function.
def shapley_value(i, characteristic_dict):
    """
    Calculate the Shapley value for a feature using the marginal gain function and the get_all_subsets function.

    Parameters
    ----------
    i : int
        Index of the feature for which to calculate the Shapley value.
    characteristic_dict : dict
        Dictionary containing characteristic values for each coalition.
    Returns
    -------
    float
        Shapley value for the feature.
    """
    F = int(log2(len(characteristic_dict)))  # Number of features
    list_of_C = np.array(get_all_subsets([i], F))
    sum = 0
    for C in list_of_C:
        cardinality = np.sum(C)  # Cardinality of the coalition
        enum = factorial(cardinality)*factorial(F - cardinality - 1)  # Number of permutations of the groups, with masked group fixed to 0
        denom = factorial(F)  # Number of permutations of the groups, with masked group fixed to 0
        normalization = enum / denom  # Normalization factor
        sum += marginal_gain(C, i, characteristic_dict)*  normalization
    return sum

def get_gt_characteristic_dict(savepath: str, env, policy_class: callable, 
                                training_function: callable, no_evaluation_episodes: int, 
                                no_train_episodes: int) -> dict:
    if exists(savepath):
        return pickle.load(open(savepath, "rb"))

    if not exists("characteristic_dicts"):
        makedirs("characteristic_dicts")

    action_space_dimension = env.action_space.n - 1
    state_feature_size = env.observation_space.shape[0]
    all_coalitions = np.array(get_all_subsets([], state_feature_size))

    def compute_characteristic(mask):
        def train_and_evaluate_single_run():
            state_space_dimension = np.sum(mask)
            policy = policy_class(state_space_dimension, action_space_dimension)
            trained_policy = training_function(policy, env, mask=mask)
            return np.mean(evaluate_policy(no_evaluation_episodes, env, trained_policy, mask))

        rewards = Parallel(n_jobs=-1)(
            delayed(train_and_evaluate_single_run)()
            for _ in range(no_train_episodes)
        )

        return (mask.tobytes(), np.mean(rewards))

    results = Parallel(n_jobs=-1)(
        delayed(compute_characteristic)(mask)
        for mask in all_coalitions
    )

    characteristic_dict = dict(results)

    pickle.dump(characteristic_dict, open(savepath, "wb"))
    return characteristic_dict








#Takes a feature i, and a mask C. Gives the marginal gain of adding feature i to the mask C, via an evaluation function
# def marginal_gain(policy, ms_ft_prd_fnc,eval_function , features, C, seed, env): 
#     """
#     Calculates the marginal gain of adding feature i to the mask C, using the eval_function.
#     """
#     C_i = np.copy(C)
#     for i in features:
#         C_i[i] = 1
# 
#     V_C = eval_function(policy, seed, ms_ft_prd_fnc, C, env)
# 
#     V_C_i= eval_function(policy, seed, ms_ft_prd_fnc, C_i, env)
# 
#     return V_C_i - V_C
# def shapley_value(policy, ms_ft_pred_fnc, eval_function,  G, masked_group, seed, env):
#     """
#     Calculate the Shapley value for a feature using the marginal gain function and the get_all_subsets function.
#     """
#     # Cardinality integers
#     num_groups = len(G) # |F| i.e. number of groups 
#     num_groups_per_C = get_r(num_groups, masked_group)  # All possible |C| for all permutations excluding the masked group
# 
#     list_of_C = get_all_group_subsets(G, masked_group, r=num_groups_per_C)
#     sum = 0
# 
#     for c, C in enumerate(list_of_C):
#         enum = (factorial(np.sum(num_groups_per_C[c]))) * factorial(num_groups - np.sum(num_groups_per_C[c]) - 1)
#         denom = factorial(num_groups)
#         normalizing_constant = enum / denom
#         marginal_gain_i = marginal_gain(policy, ms_ft_pred_fnc, eval_function, G[masked_group], C, seed, env)
#         sum += normalizing_constant * marginal_gain_i
#     return sum

