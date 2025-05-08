import gymnasium as gym 
import numpy as np
import pickle
from math import factorial
from os.path import join, exists
from os import makedirs

from sverl.cartpole_agent import PolicyCartpole, train_cartpole_agent
from sverl.sverl_utils import evaluate_policy, report_sverl_p
from sverl.group_utils import get_all_subsets

savepath = join("value_dicts", "cartpole_values_dict.pkl")
state_feature_names = ["Cart Position", "Cart Velocity", "Pole Angle", "Pole Angular Velocity"]

def marginal_gain(C, i, value_dict):

    """
    Calculate the marginal gain of adding feature i to the coalition C.
    
    Parameters
    ----------
    C : np.ndarray
        Coalition mask.
    C_i : np.ndarray
        Coalition mask with feature i added.
    value_dict : dict
        Dictionary containing values for each coalition.

    Returns
    -------
    float
        Marginal gain of adding feature i to coalition C.
    """
    C_i = np.copy(C)
    C_i[i] = 1  # Add feature i to the coalition
    V_C = value_dict[C.tobytes()]
    V_C_i = value_dict[C_i.tobytes()]
    
    return V_C_i - V_C 

all_coalitions = np.array(get_all_subsets([1], 4))

#Calculates Shapley values for a feature, using the marginal gain function and the get_all_subsets function.
def shapley_value(i, value_dict):
    """
    Calculate the Shapley value for a feature using the marginal gain function and the get_all_subsets function.
    """
    list_of_C = np.array(get_all_subsets([i], 4))
    sum = 0
    for c, C in enumerate(list_of_C):
        enum = factorial(np.sum(C))*(4 - np.sum(C) - 1)  # Number of permutations of the groups, with masked group fixed to 0
        denom = factorial(4)  # Number of permutations of the groups, with masked group fixed to 0
        normalization = enum / denom  # Normalization factor
        sum += marginal_gain(C, i, value_dict)*  normalization
    return sum

if exists(savepath):
    values_dict = pickle.load(open(savepath, "rb"))  # Load the values dictionary from file
else:
    if not exists("value_dicts"): 
        makedirs("value_dicts")

    all_coalitions = np.array(get_all_subsets([], 4))  # Get all coalitions with the first feature fixed to 0

    values_dict = {}

    for c in all_coalitions:
        values_dict[c.tobytes()] = 0  # Initialize the values for each coalition
    env = gym.make('CartPole-v1')
    for mask in all_coalitions:
        reward = 0  # Initialize reward
        no_evaluation_episodes = 10
        state_space_dimension = np.sum(mask)  # State space dimension
        action_space_dimension = env.action_space.n - 1 # This is the dimension, not the size

        for _ in range(no_evaluation_episodes):
            policy = PolicyCartpole(state_space_dimension, action_space_dimension)

            policy = train_cartpole_agent(policy, env, mask=mask)  # Train the policy

            reward += np.mean(evaluate_policy(no_evaluation_episodes, env, policy, mask)) #evaluating the policy

        values_dict[mask.tobytes()] = reward/no_evaluation_episodes  # Store the average reward for each coalition


    pickle.dump(values_dict, open(savepath, "wb"))

shapley_values = np.zeros(4)  # Initialize Shapley values for each feature
for i in range(4):
    shapley_values[i] = shapley_value(i, values_dict)  # Calculate Shapley value for each feature



report_sverl_p(shapley_values, values_dict[np.array([0, 0, 0, 0]).tobytes()], state_feature_names)