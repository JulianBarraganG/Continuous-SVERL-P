import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm, trange
from os.path import exists, join
from os import makedirs
import pickle
import time
from NeuralConditioner import NC, Discriminator, train_nc
from RandomSampler import RandomSampler
import numpy as np

class StateFeatureDataset(Dataset):
    def __init__(self, data, batch_size=32, shuffle=True):
        self.data = torch.FloatTensor(data)  # convert to pytorch tensor
        self.dataloader = DataLoader(self.data, batch_size=batch_size, shuffle=shuffle)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    
    
#gets all subsets of masks when one feature is fixed to 0.
#basically all c \in f/i 

def get_all_subsets(fixed_features, list_length):
    """
    generate all binary lists of given length with certain positions fixed to 0.
    
    args:
        fixed_features: list of indices that must be 0 in all variations
        list_length: total length of the binary lists to generate
        
    returns:
        list of all possible binary lists with the specified features fixed to 0
    """
    variations = []
    
    # calculate how many bits we need to vary (total length minus fixed positions)
    variable_positions = [pos for pos in range(list_length) if pos not in fixed_features]
    num_variable_bits = len(variable_positions)
    
    # generate all possible combinations for the variable bits
    for num in range(2 ** num_variable_bits):
        binary = [0] * list_length
        
        # fill in the variable positions
        for bit_pos in range(num_variable_bits):
            # get the current variable position in the original list
            original_pos = variable_positions[bit_pos]
            # get the bit value (0 or 1)
            bit_value = (num >> (num_variable_bits - 1 - bit_pos)) & 1
            binary[original_pos] = bit_value
            
        variations.append(binary)
    
    return variations

def get_r(k, masked_group):
    """gets the permutations of the groups, where masked group is fixed to 0."""
    return get_all_subsets([masked_group], k)

def get_all_group_subsets(g, masked_group):
    """
    gets all permutations of subsets, with masked group fixed to 0.
    args:
        g: the groups
        masked_group: the group that is fixed to 0
    returns:
        permutations: the permutations of the groups
    """

    # first index = group & second index = feature index
    n = sum(len(sublist) for sublist in g) # number of features
    k = len(g) # number of groups
    num_perms = 2**(k-1) # number of permutations / len(r)


    r = get_r(k, masked_group)

    permutations = np.ones((num_perms, n), dtype=np.int64)  # initialize the permutations array
    for l, rnoget in enumerate(r):
        p_i = np.ones(n)
        for i in range(k):
            # loop over every j feature index in the i-th group.
            for j in g[i]: 
                p_i[j] = rnoget[i]
        permutations[l] = p_i
    return permutations


def get_trajectory(policy, env, time_horizon = 10**3): 
    trajectory_features = []  # store the features of the trajectory
    state = env.reset()[0]  # forget about previous episode
    state_space_dimension = env.observation_space.shape[0]  # state space dimension
    
    
    for _ in trange(time_horizon):     

        a = policy(state)
        
        state, reward, terminated, truncated, _ = env.step(a)
        if(terminated or truncated): 
            state = env.reset()[0]
            
        trajectory_features.append(state)   
        
        
    return np.array(trajectory_features)

def save_trajectory(trajectories, filename, delimiter=','):
    """
    save the trajectory data to a csv-file.

    parameters:
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


def evaluate_policy(no_episodes, env, policy): 
    rewards = []
    for _ in trange(no_episodes):
        r = 0
        state = env.reset()[0]
        while True:  # environment sets "truncated" to true after 500 steps 
                state, reward, terminated, truncated, _ = env.step( policy(state) ) #  take a  action
                r += reward  # accumulate reward
                if terminated or truncated:
                    break
        env.close()
        rewards.append(r)
    return np.array(rewards)  # return the cumulated reward

def get_agent_and_trajectory(policy,
                             env,
                             model_filepath, 
                             trajectory_filename,
                             training_function,
                             gen_and_save_trajectory=True):
    """
    Checks if the model and trajectory files exist, otherwise trains and saves them.

    parameters:
    ----------
    policy: policy to be trained
    model_filepath: path to save the trained policy
    trajectory_filename: name of the trajectory file
    trajectory_filepath: path to save the trajectory
    training_function: function to train the policy
    env: environment to train the policy
    gen_and_save_trajectory: boolean, if true, generate and save the trajectory

    returns:
    -------
    policy: trained policy if gen_and_save_trajectory is false
    policy, trajectory: trained policy and trajectory if gen_and_save_trajectory is true
    """

    if not exists("models"):
        makedirs("models")
    if not exists(model_filepath):
        print("training agent...")
        policy = training_function(policy, env, ftarget=-9999.9) # train and report steps until convergence

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

def get_neural_conditioner(filepath, input_dim, latent_dim, dataloader):
    """
    Loads the neural conditioner from the given filepath, or creates and saves it,
    if it doesn't exist.

    parameters:
    ----------
    filepath: path to the neural conditioner model
    input_dim: input dimension of the neural conditioner
    latent_dim: latent dimension of the neural conditioner
    dataloader: dataloader for training the neural conditioner

    returns:
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
    
def get_random_sampler(filepath, trajectory):
    """
    Loads the random sampler from the given filepath, or creates it and saves it
    if it doesn't exist.

    parameters:
    ----------
    filepath: path to the neural conditioner model
    trajectory: trajectory to be used for creating the random sampler

    returns:
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
    
