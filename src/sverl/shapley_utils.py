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

def get_gt_characteristic_dict(savepath: str, env: Env, policy_class: callable, 
                                training_function: callable, no_evaluation_episodes: int, 
                                num_models: int, model_filepath: str | None = None) -> dict:
    
    """ 
    Get the dictionary of characteristic values for all coalitions based on 'ground truth' models.
    If model_filepath is not None, the policy trained on the full coalition will be saved to that path.
    This can guarantee that the ground-truth shapley values is calculated on the same 
    policy that is used when calculating shapley values via the imputation methods. 

    Parameters
    ----------
    savepath : str
    env : Env 
    policy_class : callable
        Policy class to be trained. I think this should actually be a class 
    training_function : callable
    no_evaluation_episodes : int
    num_models: int
    model_filepath : str | None            
            Path to save the policy trained on the full coalition. If None, the policy will not be saved.
    
    returns
    -------
    dict
        Dictionary of characteristic values for all coalitions.
    """

    if exists(savepath):
        return pickle.load(open(savepath, "rb"))

    if not exists("characteristic_dicts"):
        makedirs("characteristic_dicts")

    action_space_dimension = env.action_space.n - 1
    state_feature_size = env.observation_space.shape[0]
    all_coalitions = np.array(get_all_subsets([], state_feature_size))

    def compute_characteristic(mask):
        """Compute the characteristic function on avg"""
        state_space_dimension = np.sum(mask)
        policy = policy_class(state_space_dimension, action_space_dimension)
        trained_policies = [training_function(policy, env, mask=mask) for _ in range(num_models)]
        avg_performances = np.zeros(num_models)
        
        # For each trained policy, evaluate each () and pick best on avg
        for i, policy in enumerate(trained_policies):
            performance_i = evaluate_policy(no_evaluation_episodes, env, policy, mask=mask)
            avg_performances[i] = np.mean(performance_i)

        # Save model trained on the full set
        if model_filepath is not None:
            if state_space_dimension == env.observation_space.shape[0]:
                if not exists(model_filepath):
                    print(f"saving policy at: {model_filepath}")
                    best_idx = np.argmax(avg_performances)
                    best_policy = trained_policies[best_idx]
                    pickle.dump(best_policy, open(model_filepath, "wb")) #saving the policy

        best_on_avg = np.max(avg_performances)

        return (mask.tobytes(), best_on_avg)


    results = Parallel(n_jobs=-1)(
        delayed(compute_characteristic)(mask)
        for mask in all_coalitions
    )

    characteristic_dict = dict(results)

    pickle.dump(characteristic_dict, open(savepath, "wb"))
    return characteristic_dict


def get_imputed_characteristic_dict(savepath: str, env: Env, policy: callable, 
                                    imputation_fnc: callable, no_evaluation_episodes: int,
                                    char_val_fnc: callable, starting_state: np.ndarray | None = None) -> dict:
    
    """ 
    Get the dictionary of characteristic values for all coalitions, using an imputation function.

    Parameters
    ----------
    savepath : str
    env : Env
    policy : callable
    imputation_fnc : callable
    no_evaluation_episodes : int
    char_val_fnc : callable
        The characteristic value function, or "contribution function".
        This is the function input of Shapley values $\\phi_i(v)$.
    returns 
    -------
    dict
        Dictionary of characteristic values for all coalitions.
    """
    if exists(savepath):
        return pickle.load(open(savepath, "rb"))

    if not exists("characteristic_dicts"):
        makedirs("characteristic_dicts")

    action_space_dimension = env.action_space.n - 1
    state_feature_size = env.observation_space.shape[0]
    all_coalitions = np.array(get_all_subsets([], state_feature_size))

    def compute_characteristic(mask):
        reward = 0
        for seed in range(no_evaluation_episodes): 
            if starting_state != None: 
                reward += char_val_fnc(policy, starting_state, imputation_fnc, mask, env)
            else:
                reward += char_val_fnc(policy, seed, imputation_fnc, mask, env)
        return (mask.tobytes(), reward/no_evaluation_episodes)

    results = Parallel(n_jobs=-1)(
        delayed(compute_characteristic)(mask)
        for mask in all_coalitions
    )

    characteristic_dict = dict(results)

    pickle.dump(characteristic_dict, open(savepath, "wb"))
    return characteristic_dict

