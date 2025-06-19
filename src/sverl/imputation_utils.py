from contextlib import contextmanager
from gymnasium import Env
from os import makedirs
from os.path import exists, join
import pickle as pkl

import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from tqdm import trange

from .NeuralConditioner import NC, Discriminator, train_nc
from .PiSampler import PiSampler 
from vaeac.train_utils import TrainingArgs, get_vaeac
from vaeac.VAEAC import VAEAC

class StateFeatureDataset(Dataset):
    def __init__(self, data, batch_size=32, shuffle=False):
        """ 
        Converts the data to a PyTorch dataset and creates a DataLoader.
        
        Parameters
        ----------
        data : numpy.ndarray
            The data to be converted to a PyTorch dataset.
        batch_size : int, optional
            The batch size for the DataLoader. Default is 32.
        shuffle : bool, optional
            Whether to shuffle the data. Default is True.
        """
        self.data = torch.FloatTensor(data)  # convert to pytorch tensor
        self.dataloader = DataLoader(self.data, batch_size=batch_size, shuffle=shuffle)

    def __len__(self) -> int:
        """
        Returns the length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Returns the item at the specified index.
        """
        return self.data[idx]

def get_trajectory(policy: callable,
                   env: Env,
                   time_horizon: int = 10**3) -> np.ndarray:
    """
    Get a trajectory of the agent, given a policy and an environment. 
    
    Parameters
    ----------
    policy : callable
        the policy to use when generating trajectories, should take a state as input and return an action
    env : gym.Env
        the environment to generate trajectories from
    time_horizon : int
        the time horizon (T) to generate trajectories for, default is 10**3
    
    Returns
    -------
    trajectory_features: numpy.ndarray
        the trajectory features, shape (t, d + 1), where t is the time horizon and d is the state space dimension
    """

    state = env.reset()[0]  # forget about previous episode, and sample s_0 ~ p_0 
    state_space_dim = env.observation_space.shape[0]  # type: ignore
    trajectory = np.zeros((time_horizon, state_space_dim + 1))  # create a zero vector of the state space dimension
    
    for t in trange(time_horizon):     

        a = policy(state)
        trajectory[t, :-1] = state
        trajectory[t, -1] = a
        
        state, _ , terminated, truncated, _ = env.step(a)
        if(terminated or truncated): 
            state = env.reset()[0]

    return trajectory

def save_trajectory(trajectories: np.ndarray,
                    filename: str,
                    delimiter: str = ','):
    """
    Save the trajectory data to a csv-file.

    Parameters
    ----------
    trajectories: numpy array
    the trajectories to save,
    usually shape (t, d)
    filename: str
        note that 'filename' should not include document type i.e. '.csv'
    """

    # if the directory does not exist, create it
    if not exists('data'):
        makedirs('data')
    # get the file type based on the delimiter
    if delimiter == ',':
        file_type = '.csv'
    elif delimiter == '\t':
        file_type = '.tsv'
    else:
        file_type = '.txt'
    # save as csv file
    save_path = filename + file_type
    file_path = join('data', save_path)
    np.savetxt(file_path, trajectories, delimiter=delimiter)

def load_neural_conditioner(filepath: str,
                            input_dim: int | None = None, 
                            latent_dim: int | None = None,
                            dataloader: DataLoader | None = None) -> NC:
    """
    Loads the neural conditioner from the given filepath, or creates and saves it,
    if it doesn't exist.

    Parameters
    ----------
    filepath : str 
    input_dim : int 
    latent_dim : int 
    dataloader : DataLoader

    Returns
    -------
    NC
        Loaded neural conditioner model
    """
    if not exists("imputation_models"):
        makedirs("imputation_models")

    # If pkl doesn't exist, input params should be 
    if not exists(filepath) and latent_dim != None:
        nc = NC(input_dim, latent_dim)
        discriminator = Discriminator(input_dim)
        print("Training Neural Conditioner...")
        train_nc(nc, discriminator, dataloader, epochs=10)
        with open(filepath, "wb") as f:
            pkl.dump(nc, f) #saving the neural conditioner
        print(f"Neural Conditioner saved at: {filepath}")
        return nc
    else: 
        print("Loading Neural Conditioner...")
        with open(filepath, "rb") as f:
            nc = pkl.load(f)
        return nc
    
def load_random_sampler(filepath, trajectory=None):
    """
    Loads the random sampler from the given filepath, or creates it and saves it
    if it doesn't exist.

    Parameters
    ----------
    filepath: path to the neural conditioner model
    trajectory: trajectory to be used for creating the random sampler

    Returns
    -------
    nc: loaded random sampling model
    """
    if not exists("imputation_models"):
        makedirs("imputation_models")

    if (not exists(filepath)) and (trajectory is not None):
        print("Training Pi Sampler...")
        rs = PiSampler(trajectory)
        with open(filepath, "wb") as f:
            pkl.dump(rs, f) #saving the random sampler
        print(f"Pi Sampler saved at: {filepath}")
        return rs
    else: 
        print("Loading Pi Sampler...")
        with open(filepath, "rb") as f:
            rs = pkl.load(f)
        return rs
    
def load_vaeac(savepath: str, 
              data: str | np.ndarray | None = None,
              args: TrainingArgs | None = None,
              one_hot_max_sizes: list | None = None,
              nn_size_dict: dict | None = None) -> VAEAC:
    """
    Given a specification on state feature data types, and trajectory data file path,
    this function will train a VAEAC model on the data, and return the VAEAC model.

    Parameters
    ----------
    savepath : str
    data : str | np.ndarray | None
    args : TrainArgs | None
    one_hot_max_sizes : list | None

    Returns
    -------
    VAEAC : VAEAC
        The trained VAEAC model as a PyTorch module.
    """
    if not exists("imputation_models"):
        makedirs("imputation_models")

    if not exists(savepath):
        print("Training VAEAC...")
        vaeac = get_vaeac(args, one_hot_max_sizes, data, nn_size_dict)
        with open(savepath, "wb") as f:
            pkl.dump(vaeac, f)
        print(f"VAEAC saved at: {savepath}")
        return vaeac
    else:
        print("Loading VAEAC...")
        with open(savepath, "rb") as f:
            vaeac = pkl.load(f)
        return vaeac

def get_policy_and_trajectory(policy: callable,
                              env: Env,
                              model_filepath: str,
                              trajectory_filename: str,
                              training_function: callable,
                              gen_and_save_trajectory: bool = True, 
                              no_evaluation_episodes = 100, 
                              no_states_in_trajectories = 10**6):
    """
    Checks if the model and trajectory files exist, otherwise trains and saves them.
    NB: that this function will always return the policy,
    as in: $$\\pi : \\mathcal{S} \\rightarrow \\mathcal{A}$$
    So often it will be the prediction method associated with the class,
    representing the function approximation of the policy -- e.g. a Neural Network.

    Parameters
    ----------
    policy : policy to be trained
    model_filepath : path to save the trained policy
    trajectory_filename : name of the trajectory file
    trajectory_filepath : path to save the trajectory
    training_function : function to train the policy
        This function should return $\\pi$, 
        i.e. function that takes a state and returns action
    env : environment to train the policy
    gen_and_save_trajectory : boolean, if true, generate and save the trajectory

    Returns
    -------
    policy : trained policy if gen_and_save_trajectory is false
    policy, trajectory : trained policy and trajectory if gen_and_save_trajectory is true
    """

    if not exists("models"):
        makedirs("models")
    if not exists(model_filepath):
        print("training agent...")
        policy = training_function(policy, env) # train and report steps until convergence

        # check that policy learned
        print("evaluating policy...")
        no_evaluation_episodes = no_evaluation_episodes
        reward = evaluate_policy(no_evaluation_episodes, env, policy) #evaluating the policy
        print("average reward when running ", no_evaluation_episodes, " episodes: ", np.mean(reward)) #printing the average reward
        print("standard deviation when running ", no_evaluation_episodes, " episodes:: ", np.std(reward)) #printing the standard deviation of the reward
        # save the policy
        print(f"saving policy at: {model_filepath}")
        with open(model_filepath, "wb") as f:
            pkl.dump(policy, f) # saving policy
        # generate and save trajectory
        if gen_and_save_trajectory:
            print("generating trajectory...")
            trajectory = get_trajectory(policy, env, time_horizon = no_states_in_trajectories) #running the agent for 20 times, and storing the results
            # save trajectory to a csv file
            save_trajectory(trajectory, trajectory_filename, delimiter=',') # saved at data/cartpole_trajectory.csv
            print("trajectory saved at: ", trajectory_filename)
            return policy, trajectory
    else: # if the model already exists, we load it and the trajectory
        print("loading agent...")
        with open(model_filepath, "rb") as f:
            policy = pkl.load(f) #loading the policy

        print("loading trajectory...")
        if gen_and_save_trajectory:
            print("generating trajectory...")
            trajectory = get_trajectory(policy, env, time_horizon = no_states_in_trajectories) #running the agent for 20 times, and storing the results
            # save trajectory to a csv file
            save_trajectory(trajectory, trajectory_filename, delimiter=',') # saved at data/cartpole_trajectory.csv
            print("trajectory saved at: ", trajectory_filename)
            return policy, trajectory

    return policy, None

def evaluate_policy(no_episodes: int,
                    env: Env,
                    policy: callable,
                    mask: list[int]|None = None,
                    verbose: bool = False, 
                    reset_seed : int | None = None) -> np.ndarray: 
    """
    Evaluate the policy by running it in the environment for a number of episodes.

    Parameters
    ----------
    no_episodes : int
    env : Env
    policy : callable
    reset_seed : int

    Returns
    -------
    rewards : numpy.ndarray
        The rewards obtained by the policy in each episode.
    """
    # rewards = []
    # if mask is not None:
    #     mask = np.array(mask).astype(bool)
    # # Check and see what type (np.ndarray or torch.Tensor) policy expects the state to be
    # try:
    #     dummy_state = env.reset()[0]
    #     policy(torch.as_tensor(dummy_state))
    #     expects_torch = True
    # except:
    #     expects_torch = False

    # Set an environment reset seed. We want unbiased estimate of policy.
    # I.e. s_0 is always the same, and not sampled from p_0

    # Pre-compute mask and tensor conversion requirements
    mask = mask.astype(bool) if mask is not None else None # type: ignore
    expects_torch = hasattr(policy, '__code__') and 'torch' in policy.__code__.co_names
    
    # Pre-allocate rewards array
    rewards = np.empty(no_episodes, dtype=np.float32)
    
    # Cache environment reset and step methods
    reset_fn = env.reset
    step_fn = env.step
    
    # Determine state processing function
    def process_state(state):
        if mask is not None:
            state = state[mask]
        return torch.as_tensor(state) if expects_torch else state
    
    eval_range = trange(no_episodes, desc="Evaluating Policy") if verbose else range(no_episodes)
    
    for i in eval_range:
        episode_reward = 0.0
        state = process_state(reset_fn(seed=reset_seed)[0]) if reset_seed != None else process_state(reset_fn()[0])
        
        while True:
            action = policy(state)
            state, reward, terminated, truncated, _ = step_fn(action)
            state = process_state(state)
            episode_reward += float(reward)
            
            if terminated or truncated:
                break
                
        rewards[i] = episode_reward
    
    env.close()
    if verbose:
        print(f"Average reward over {no_episodes} episodes: {np.mean(rewards):2f}")
        print(f"Standard deviation of rewards: {np.std(rewards):.2f}")
    return rewards

