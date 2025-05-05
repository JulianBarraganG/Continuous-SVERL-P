import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import trange
from os.path import exists, join
from os import makedirs
import gym
import pickle

from .NeuralConditioner import NC, Discriminator, train_nc
from .RandomSampler import RandomSampler
from .shapley import shapley_value, global_sverl_value_function
from vaeac.train_utils import TrainingArgs, get_vaeac

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

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns the item at the specified index.
        """
        return self.data[idx]

def get_trajectory(policy, env, time_horizon = 10**3): 
    """
    Get a trajectory of the agent, given a policy and an environment. 
    
    Parameters
    ----------
    policy: callable
        the policy to use when generating trajectories, should take a state as input and return an action
    env: gym.Env
        the environment to generate trajectories from
    time_horizon: int
        the time horizon to generate trajectories for, default is 10**3
    
    Returns
    -------
    trajectory_features: numpy.ndarray
        the trajectory features, shape (t, d), where t is the time horizon and d is the state space dimension
    """
    trajectory_features = []  # store the features of the trajectory
    state = env.reset()[0]  # forget about previous episode    
    
    for _ in trange(time_horizon):     

        a = policy(state)
        
        state, _ , terminated, truncated, _ = env.step(a)
        if(terminated or truncated): 
            state = env.reset()[0]
            
        trajectory_features.append(state)          
        
    return np.array(trajectory_features)

def evaluate_policy(no_episodes: int, env: gym.Env, policy: callable) -> np.ndarray: 
    """
    Evaluate the policy by running it in the environment for a number of episodes.

    Parameters
    ----------
    no_episodes : int
    env : gym.Env
    policy : callable

    Returns
    -------
    rewards : numpy.ndarray
        The rewards obtained by the policy in each episode.
    """
    rewards = []
    # Check and see what type (np.ndarray or torch.Tensor) policy expects the state to be
    try:
        dummy_state = env.reset()[0]
        policy(torch.as_tensor(dummy_state))
        expects_torch = True
    except:
        expects_torch = False

    for _ in trange(no_episodes):
        r = 0
        state = env.reset()[0]
        state = torch.as_tensor(state) if expects_torch else state
        while True:  # environment sets "truncated" to true after 500 steps 
                state, reward, terminated, truncated, _ = env.step( policy(state) ) #  take a  action
                state = torch.as_tensor(state) if expects_torch else state
                r += reward  # accumulate reward
                if terminated or truncated:
                    break
        env.close()
        rewards.append(r)
    return np.array(rewards)  # return the cumulated reward

def save_trajectory(trajectories, filename, delimiter=','):
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

def get_policy_and_trajectory(policy,
                             env,
                             model_filepath, 
                             trajectory_filename,
                             training_function,
                             gen_and_save_trajectory=True):
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
        no_evaluation_episodes = 100
        reward = evaluate_policy(no_evaluation_episodes, env, policy) #evaluating the policy
        print("average reward when running ", no_evaluation_episodes, " episodes: ", np.mean(reward)) #printing the average reward
        print("standard deviation when running ", no_evaluation_episodes, " episodes:: ", np.std(reward)) #printing the standard deviation of the reward
        # save the policy
        print(f"saving policy at: {model_filepath}")
        pickle.dump(policy, open(model_filepath, "wb")) #saving the policy
        # generate and save trajectory
        if gen_and_save_trajectory:
            print("generating trajectory...")
            trajectory = get_trajectory(policy, env, time_horizon = 10**5) #running the agent for 20 times, and storing the results
            # save trajectory to a csv file
            save_trajectory(trajectory, trajectory_filename, delimiter=',') # saved at data/cartpole_trajectory.csv
            print("trajectory saved at: ", trajectory_filename)
            return policy, trajectory
    else: # if the model already exists, we load it and the trajectory
        print("loading agent...")
        policy = pickle.load(open(model_filepath, "rb")) #loading the policy
        print("loeading trajectory...")
        if gen_and_save_trajectory:
            trajectory_filepath = join("data", trajectory_filename + ".csv")
            trajectory = np.loadtxt(trajectory_filepath, delimiter=",") #loading the trajectory
            return policy, trajectory

    return policy, None

def load_neural_conditioner(filepath, input_dim=None, latent_dim=None, dataloader=None):
    """
    Loads the neural conditioner from the given filepath, or creates and saves it,
    if it doesn't exist.

    Parameters
    ----------
    filepath: path to the neural conditioner model
    input_dim: input dimension of the neural conditioner
    latent_dim: latent dimension of the neural conditioner
    dataloader: dataloader for training the neural conditioner

    Returns
    -------
    nc: loaded neural conditioner model
    """
    if not exists("imputation_models"):
        makedirs("imputation_models")

    if not exists(filepath):
        nc = NC(input_dim, latent_dim)
        discriminator = Discriminator(input_dim)
        print("Training Neural Conditioner...")
        train_nc(nc, discriminator, dataloader, epochs=10)
        pickle.dump(nc, open(filepath, "wb")) #saving the neural conditioner
        print(f"Neural Conditioner saved at: {filepath}")
        return nc
    else: 
        print("Loading Neural Conditioner...")
        nc = pickle.load(open(filepath, "rb"))
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

    if not exists(filepath):
        print("Training Random Sampler...")
        rs = RandomSampler(trajectory)
        pickle.dump(rs, open(filepath, "wb")) #saving the random sampler
        print(f"Random Sampler saved at: {filepath}")
        return rs
    else: 
        print("Loading Random Sampler...")
        rs = pickle.load(open(filepath, "rb"))
        return rs
    
def load_vaeac(savepath: str, 
              data: str | np.ndarray | None = None,
              args: TrainingArgs | None = None,
              one_hot_max_sizes: list | None = None):
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
        vaeac = get_vaeac(args, one_hot_max_sizes, data)
        pickle.dump(vaeac, open(savepath, "wb"))
        print(f"VAEAC saved at: {savepath}")
        return vaeac
    else:
        print("Loading VAEAC...")
        vaeac = pickle.load(open(savepath, "rb"))
        return vaeac

def get_sverl_p(policy,
                env: gym.Env,
                feature_imputation_fnc: callable,
                G: list | None = None,
                num_rounds: int = 10,
                starting_state: np.ndarray | None = None,
                characteristic_fnc: callable = global_sverl_value_function):
    """
    Calculate the SVERL-P for the given policy and environment.
    groupShapley can be calculated, by specifying the groups in G.
    If G is None, then individual Shapley values are calculated.
    Parameters
    ----------
    policy : callable 
        $\\pi : \\mathcal{S} \\rightarrow \\mathcal{A}$
    env : gym.Env
    feature_imputation_fnc : callable
    G : list | None
        Groups of features. If None, then individual Shapley values are calculated.
    num_rounds : int
    starting_state : np.ndarray | None
        For local SVERL-P.
    Returns
    -------
    shapley_values : numpy.ndarray
        SVERL-P values for each (group of) feature(s).
    value_empty_set : float

    TODO: Can't currently calculate local SVERL-P.
    """

    num_features = env.observation_space.shape[0]
    if G is None: # No grouping
        G = [[i] for i in range(num_features)]
    value_empty_set_sum = 0
    shapley_values_sum = np.zeros(num_features)
    for i in trange(num_rounds): 
        value_empty_set_sum += characteristic_fnc(policy, i, feature_imputation_fnc, np.zeros(4), env)
        for g in range(len(G)):
            shapley_values_sum[g] += shapley_value(policy, feature_imputation_fnc, characteristic_fnc, G, g, i, env)

    shapley_values = shapley_values_sum / num_rounds
    value_empty_set = value_empty_set_sum / num_rounds

    return shapley_values, value_empty_set

def report_sverl_p(shap_vls: list,
                    vl_empty_set: list,
                    state_feature_names: list) -> None:
    """
    Report the Shapley values of the state features and the value of the empty set.
    """
    right_adjust = max([(len(f_name)) for f_name in state_feature_names]) + 1
    empty_set_prefix = "Value of empty set"
    prefix = "Shapley value of "

    for i, shap_vl in enumerate(shap_vls):
        print(f"Shapley value of {state_feature_names[i]:<{right_adjust}}: {shap_vl:>8.2f}")
    print(f"{empty_set_prefix:<{right_adjust + len(prefix)}}: {vl_empty_set:>8.2f}")

