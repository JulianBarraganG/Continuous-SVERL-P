from gymnasium import Env
from math import factorial, log2
import numpy as np

from collections import defaultdict
from joblib import Parallel, delayed
from os.path import exists
from os import makedirs
import pickle
import tqdm as tqdm
from operator import itemgetter

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

    action_space_dim = env.action_space.n - 1
    state_feature_size = env.observation_space.shape[0]
    all_coalitions = np.array(num_models * get_all_subsets([], state_feature_size))
    reset_seed = 42

    def process_task(mask):
        """Process a single (coalition, model) training/evaluation task"""
        policy = policy_class(mask.sum(), action_space_dim)
        trained_policy = training_function(policy, env, mask=mask)
        performance = evaluate_policy(no_evaluation_episodes, env, trained_policy,
                                      mask=mask)
        mean_perf = np.mean(performance)
        seed_perf = evaluate_policy(1, env, trained_policy,
                                      mask=mask, reset_seed=reset_seed)

        
        # Track if we need to save this policy (full coalition only)
        save_policy = (model_filepath is not None) and mask.all()
        return (
            mask.tobytes(),
            mean_perf,
            seed_perf,
            trained_policy if save_policy else None
        )

    # Execute all tasks in parallel
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(process_task)(coalition)
        for coalition in all_coalitions
    )

    # Aggregate results and find maximum performance per coalition
    mask_performances = defaultdict(list)
    full_coalition_candidates = []

    for mask_bytes, mean_perf, seed_perf, policy in results:
        mask_performances[mask_bytes].append(mean_perf, seed_perf)
        if policy is not None:
            full_coalition_candidates.append((mean_perf, seed_perf, policy))

    # Create final characteristic dictionary
    characteristic_dict = {
        mask: max(tuple_list, key = itemgetter(0))[1] # for key C, saves seed_perf with max(mean_perf) in (max_perf, seed_perf)
        for mask, tuple_list in mask_performances.items()
    }

    # Save best policy for full coalition if needed
    if model_filepath and full_coalition_candidates:
        best_perf, best_policy = max(full_coalition_candidates, key=lambda x: x[0])
        if not exists(model_filepath):
            if not exists("models"):
                makedirs("models")
            print(f"Saving best policy to {model_filepath}")
            pickle.dump(best_policy, open(model_filepath, "wb"))

    # Save results and return
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
        for _ in range(no_evaluation_episodes): 
            if starting_state != None: # Local SVERL
                reward += char_val_fnc(policy, starting_state, imputation_fnc, mask, env)
            else:                      # Global SVERL
                reward += char_val_fnc(policy, imputation_fnc, mask, env)
        return (mask.tobytes(), reward/no_evaluation_episodes)

    results = Parallel(n_jobs=-1)(
        delayed(compute_characteristic)(mask)
        for mask in all_coalitions
    )

    characteristic_dict = dict(results)

    pickle.dump(characteristic_dict, open(savepath, "wb"))
    return characteristic_dict

