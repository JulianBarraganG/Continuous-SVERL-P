import gymnasium as gym 
import numpy as np
import pickle
from math import factorial, log2
from os.path import join, exists
from os import makedirs

from sverl.cartpole_agent import PolicyCartpole, train_cartpole_agent
from sverl.sverl_utils import evaluate_policy, report_sverl_p
from sverl.group_utils import get_all_subsets

savepath = join("characteristic_dicts", "gt_cartpole_characteristic_dict.pkl")
state_feature_names = ["Cart Position", "Cart Velocity", "Pole Angle", "Pole Angular Velocity"]

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


def get_characteristic_dict(savepath: str, env: gym.Env, policy_class: callable, 
                            training_function: callable, no_evaluation_episodes: int , 
                            no_train_episodes: int) -> dict:
    """
    Get the characteristic dictionary for the given environment and policy class.
    
    Parameters
    ----------
    savepath : str
    env: gym.Env
    policy_class : callable
    training_function : callable
    no_evaluation_episodes : int
        Number of evaluation episodes per trained policy.
    no_train_episodes : int
        Number of polcies to be trained for each coalition.

    Returns
    -------
    dict
    """
    if exists(savepath):
        characteristic_dict = pickle.load(open(savepath, "rb"))  # Load the characteristic dictionary from file
        return characteristic_dict
    else:
        if not exists("characteristic_dicts"): 
            makedirs("characteristic_dicts")
        action_space_dimension = env.action_space.n - 1 # This is the dimension, not the size
        state_feature_size = env.observation_space.shape[0]  # Get the size of the state space
        all_coalitions = np.array(get_all_subsets([], state_feature_size))  # Get all coalitions with the first feature fixed to 0

        characteristic_dict = {}

        for c in all_coalitions:
            characteristic_dict[c.tobytes()] = 0  # Initialize the characteristic for each coalition

        for mask in all_coalitions:
            for _ in range(no_train_episodes):
                reward = 0  # Initialize reward
                state_space_dimension = np.sum(mask)  # State space dimension            

                policy = policy_class(state_space_dimension, action_space_dimension)

                policy = training_function(policy, env, mask=mask)  # Train the policy

                reward += np.mean(evaluate_policy(no_evaluation_episodes, env, policy, mask)) #evaluating the policy

            characteristic_dict[mask.tobytes()] = reward/no_train_episodes  # Store the average reward for each coalition


        pickle.dump(characteristic_dict, open(savepath, "wb"))
        return characteristic_dict


characteristic_dict = get_characteristic_dict(savepath, gym.make("CartPole-v1"), PolicyCartpole, train_cartpole_agent, 10, 10)  # Get the characteristic dictionary

shapley_values = np.zeros(4)  # Initialize Shapley values for each feature
for i in range(4):
    shapley_values[i] = shapley_value(i, characteristic_dict)  # Calculate Shapley value for each feature



report_sverl_p(shapley_values, characteristic_dict[np.array([0,0,0,0]).tobytes()], 
               characteristic_dict[np.array([1,1,1,1]).tobytes()], state_feature_names)