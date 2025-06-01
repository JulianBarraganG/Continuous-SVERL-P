import csv
from gymnasium import Env
import numpy as np
import os

def local_sverl_value_function(policy, initial_state, imputation_fnc, mask, env):
    """
    Evaluate the policy from a given state, using the believed state to make the initial decision
    Parameters
    ----------
    policy : function
    initial_state : numpy.ndarray
    imputation_fnc : function
    mask : np.ndarray
    env : gym.Env
    Returns
    -------
    R : float
        The cumulated reward
    """
    R = 0
    
    
    believed_initial_state = imputation_fnc(initial_state, mask)
    env.reset() 
    env.unwrapped.state = initial_state

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

def global_sverl_value_function(policy: callable, 
                                imputation_fnc: callable,
                                mask: np.ndarray, env: Env
                                ) -> float:
    """
    Evaluate the policy with missing features, using the believed state to make the decision.
    Parameters
    ----------
    policy : callable
    imputation_fnc : callable
    mask : np.ndarray
    env : gym.Env
    Returns
    -------
    R : float
        The cumulated reward
    """
    R = 0.
    seed = 42 # should be global, but 42 here and in evaluate policy
    true_state = env.reset(seed=seed)[0]  # Forget about previous episode
    
    believed_state = imputation_fnc(true_state.flatten(), mask)  

    while True:     
        a = policy(believed_state)
        
        state, reward, terminated, truncated, _ = env.step(a)  # Simulate pole
        R += float(reward)
        believed_state = imputation_fnc(state.flatten(), mask)
       
        if(terminated or truncated): 
            break
        
    env.close()
    return R

def report_sverl_p(shapley_values: np.ndarray,
                    state_feature_names: list[str],
                   data_file_name: str = "",
                    row_name: str | None = None,
                   verbose: bool = True) -> None:
    """
    Report the Shapley values of the state features and the value of the empty set.

    Parameters
    ----------
    shapley_values : list
        Shapley values for each state feature.
    state_feature_names : list[str]
        Names of the state features.
    data_file_name : str
        File will be saved at data_file_name.csv
    row_name : str | None
    verbose : bool (default true)
    """
    csv_save_name = data_file_name + ".csv"

    # Print the Shapley values.
    if verbose:
        right_adjust = max([(len(f_name)) for f_name in state_feature_names]) + 1
        for i, shap_vl in enumerate(shapley_values):
            print(f"Shapley value of {state_feature_names[i]:<{right_adjust}}: {shap_vl:>8.2f}")

    # Write to csv file.
    if row_name is not None:
        if data_file_name == "":
            raise ValueError("data_file_name must be specified if row_name is provided.")
        if not os.path.exists("data"):
            os.makedirs("data")
        file_path = os.path.join("data", csv_save_name)
        row = [row_name] + shapley_values.tolist()
        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)

