from gymnasium import make
import numpy as np
from os.path import join, exists

from sverl.cartpole_agent import PolicyCartpole, train_cartpole_agent
from sverl.sverl_utils import report_sverl_p
from sverl.shapley_utils import get_gt_characteristic_dict, shapley_value

def get_gt_cartpole(num_eval_eps: int = 100, num_models: int = 16):
    """
    Get GT Cartpole estimations.

    Parameters
    ----------
    num_eval_eps : int
    num_models : int

    Returns
    -------
    shapley_values : np.ndarray
    """
    # Instantiating variables
    env = make("CartPole-v1")
    savepath = join("characteristic_dicts", "gt_cartpole_characteristic_dict.pkl")
    model_filepath = join("models", "cartpole_policy.pkl")
    state_space_dim = env.observation_space.shape[0] # State space dimension #type: ignore
    shapley_values = np.zeros(state_space_dim)  # Initialize Shapley values for each feature

    # Get the ground truth characteristic dictionary
    if not exists(model_filepath):
        print(f"Evaluating {num_models} models with {num_eval_eps} evaluation episodes each, for GT Cartpole...")
    characteristic_dict = get_gt_characteristic_dict(savepath, env, PolicyCartpole, train_cartpole_agent,
                                                     num_eval_eps, num_models, model_filepath)

    # Calculate the Shapley values for each feature
    for i in range(state_space_dim):
        shapley_values[i] = shapley_value(i, characteristic_dict)  # Calculate Shapley value for each feature

    # Empty set available, not currently used
    # empty_set_mask = np.array([0,0,0,0]).tobytes()
    # empty_set_val = characteristic_dict[empty_set_mask]  # Get the value of the empty set

    return shapley_values

if __name__ == "__main__":
    print("Running 'gt_cartpole.py' directly. Purely for testing.")
    get_gt_cartpole(num_eval_eps=1, num_models=1) 

