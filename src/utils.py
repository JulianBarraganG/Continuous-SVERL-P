import gymnasium as gym
import numpy as np
import pickle as pkl
from os.path import exists, join

from sverl.imputation_utils import get_trajectory, save_trajectory



def get_supervised_learning_data(C: list[int],
                                 env: gym.Env,
                                 model_filepath: str,
                                 trajectory_filename: str,
                                 T: int,
                                 include_val: bool = False,
                                 train_perc: float = 0.8) -> tuple[np.ndarray,...]:
    """
    Get train, test split of supervised learning data from the CartPole environment.
    Validation set can be included if specified.
    If the policy model does not exist, it raises a ValueError.
    If the trajectory file does not exist, it generates a new trajectory and saves it.
    
    Parameters
    ----------
    C : list[int]
        A binary masks. Determines coalition of state features.
    env : gym.Env
    model_filepath : str
        The file path to the policy model.
    trajectory_filename : str
    T : int
        The number of timesteps to generate in the trajectory.
    include_val : bool
        Whether to include a validation set in the split.
    train_perc : float
        The percentage of data to use for training if no validation set is included.

    Returns
    -------
    tuple[np.ndarray, ...]
        The train and test (and validation) splits of the data.
        If include_val is True, returns (X_train, y_train, X_val, y_val, X_test, y_test).
        Otherwise, returns (X_train, y_train, X_test, y_test).
    """
    # Load the environment and policy
    state_space_dim = env.observation_space.shape[0] # type: ignore
    assert (len(C) == state_space_dim) or (len(C) == state_space_dim + 1), \
    "Coalition mask length must match number of state features (or one more for actions)"
    if len(C) == state_space_dim:
        # Coalition given without actions. Coalition never includes actions.
        C.append(0) 

    # Check for model file existence
    if not exists(model_filepath):
        raise ValueError("Get Pi_C function assumes model exists")

    with open(model_filepath, "rb") as f:
        policy =  pkl.load(f)

    # Check if the trajectory file exists, if not, generate it
    csv_file = join("data", trajectory_filename + ".csv")
    if not exists(csv_file):
        trajectory = get_trajectory(policy, env, time_horizon=T)
        save_trajectory(trajectory, trajectory_filename)
    else:
        trajectory = np.loadtxt(csv_file, delimiter=",")
    
    assert trajectory.shape[1] == state_space_dim + 1, "Trajectory data must have one more column than state features (for actions)"

    # Get the data
    coalition_indices = np.array(C, dtype=bool) # Converts boolean mask to indices
    X = trajectory[:, coalition_indices]
    y = trajectory[:, -1].astype(int) # Binary actions

    # Shuffle the data 
    permutation = np.random.permutation(T)
    X = X[permutation]
    y = y[permutation]

    if include_val:
        # Split the data into train, validation, and test sets
        sp = 0.6
        train_split = int(T * sp) # 60% train
        val_split = int(T * (sp + 0.2)) # 20% validation and 20% test
        X_train, X_val, X_test = X[:train_split], X[train_split:val_split], X[val_split:]
        y_train, y_val, y_test = y[:train_split], y[train_split:val_split], y[val_split:]

        return X_train, y_train, X_val, y_val, X_test, y_test

    # If no validation set, just train and test
    sp = 0.8 # sp % train, (1 - sp) % test
    train_split = int(T * sp)
    X_train, X_test = X[:train_split], X[train_split:]
    y_train, y_test = y[:train_split], y[train_split:]

    return X_train, y_train, X_test, y_test

