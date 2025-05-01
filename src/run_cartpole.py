import gymnasium as gym  # Defines RL environments
import numpy as np  # For numerical operations
from os.path import join, exists
from tqdm import trange
from torch.utils.data import Dataset, DataLoader

import sverl.shapley as shapley
from sverl.utils import StateFeatureDataset, get_agent_and_trajectory, load_neural_conditioner, load_random_sampler, load_vaeac
from sverl.cartpole_agent import PolicyCartpole, train_cartpole_agent

import vaeac.VAEAC as VAEAC
from vaeac.VAEAC_train import get_vaeac
from vaeac.train_utils import TrainingArgs

env = gym.make('CartPole-v1')
state_space_dimension = env.observation_space.shape[0]

action_space_dimension = env.action_space.n - 1

policy = PolicyCartpole(state_space_dimension, action_space_dimension)
model_filepath = join("models", "cartpole_policy.pkl")
trajectory_filename = "cartpole_trajectory"
rs_filepath = join("imputation_models", "cartpole_rs.pkl")
nc_filepath = join("imputation_models", "cartpole_nc.pkl")
vaeac_filepath = join("imputation_models", "cartpole_vaeac.pkl")

# Check if the model and trajectory files exist, otherwise train and save them
feature_imputation_model_missing = not(exists(rs_filepath) 
                                       and exists(nc_filepath) 
                                       and exists(vaeac_filepath))

# We assume trajectory file is csv
policy, trajectory = get_agent_and_trajectory(policy,
                                              env,
                                              model_filepath,
                                              trajectory_filename,
                                              train_cartpole_agent,
                                              gen_and_save_trajectory=feature_imputation_model_missing)

if feature_imputation_model_missing:
    batch_size = 32
    dataset = StateFeatureDataset(trajectory, batch_size=batch_size, shuffle=True)
    dataloader = dataset.dataloader
    input_dim = 4  # size of the data
    latent_dim = 64  # Size of the latent space

    nc = load_neural_conditioner(nc_filepath, input_dim=input_dim, latent_dim=latent_dim, dataloader=dataloader)
    rs = load_random_sampler(rs_filepath, trajectory=trajectory)
    vaeac = load_vaeac(vaeac_filepath, data=trajectory, args=TrainingArgs(), one_hot_max_sizes=[0,0,0,0])
else: 
    nc = load_neural_conditioner(nc_filepath)
    rs = load_random_sampler(rs_filepath)
    vaeac = load_vaeac(vaeac_filepath)


#The i is the seed. This is the only way I know how to set the starting position 
#We are doing 100 different seeds, and averaging the results.
NUM_ROUNDS = 10

shapley_cart_pos = 0
shapley_cart_vel = 0
shapley_pole_angle = 0
shapley_pole_vel = 0
value_empty_set = 0
G= [[0], [1], [2], [3]] #The groups of features. In this case, we have 4 features, and each feature is its own group.
print("Calculating Shapley values based on Random Sampler...")
for i in trange(NUM_ROUNDS): 
    value_empty_set += shapley.global_sverl_value_function(policy, i, rs.pred, np.zeros(4), env)
    shapley_cart_pos += shapley.shapley_value(policy, rs.pred, shapley.global_sverl_value_function, G, 0, i, env)
    shapley_cart_vel += shapley.shapley_value(policy, rs.pred, shapley.global_sverl_value_function, G, 1, i, env)
    shapley_pole_angle += shapley.shapley_value(policy, rs.pred, shapley.global_sverl_value_function, G, 2, i, env)
    shapley_pole_vel += shapley.shapley_value(policy, rs.pred, shapley.global_sverl_value_function, G, 3, i, env)
shapley_cart_pos /= NUM_ROUNDS
shapley_cart_vel /= NUM_ROUNDS
shapley_pole_angle /= NUM_ROUNDS
shapley_pole_vel /= NUM_ROUNDS
value_empty_set /= NUM_ROUNDS

print("Shapley value of Cart Position: ", shapley_cart_pos)
print("Shapley value of Cart Velocity: ", shapley_cart_vel)
print("Shapley value of Pole Angle: ", shapley_pole_angle)
print("Shapley value of Pole Angular Velocity: ", shapley_pole_vel)
print("Value of empty set: ", value_empty_set)



NUM_ROUNDS = 10
shapley_pos = 0
shapley_vel = 0
value_empty_set = 0
G= [[0,2], [1,3]] #The groups of features. In this case, we have 4 features, and each feature is its own group.
print("Calculating Shapley values based on Random Sampler...")
for i in trange(NUM_ROUNDS): 
    value_empty_set += shapley.global_sverl_value_function(policy, i, rs.pred, np.zeros(4), env)
    shapley_pos += shapley.shapley_value(policy, rs.pred, shapley.global_sverl_value_function, G, 0, i, env)
    shapley_vel += shapley.shapley_value(policy, rs.pred, shapley.global_sverl_value_function, G, 1, i, env)
shapley_pos /= NUM_ROUNDS
shapley_vel /= NUM_ROUNDS
value_empty_set /= NUM_ROUNDS

print("Shapley value of cart an angular position: ", shapley_pos)
print("Shapley value of cart and angular velocity: ", shapley_vel)
print("Value of empty set: ", value_empty_set)



#The i is the seed. This is the only way I know how to set the starting position 
#We are doing 100 different seeds, and averaging the results.
NUM_ROUNDS = 10

shapley_cart_pos = 0
shapley_cart_vel = 0
shapley_pole_angle = 0
shapley_pole_vel = 0
value_empty_set = 0
G= [[0], [1], [2], [3]] #The groups of features. In this case, we have 4 features, and each feature is its own group.
print("Calculating Shapley values based on NC...")
for i in trange(NUM_ROUNDS): 
    value_empty_set += shapley.global_sverl_value_function(policy, i, nc.pred, np.zeros(4), env)
    shapley_cart_pos += shapley.shapley_value(policy, nc.pred, shapley.global_sverl_value_function, G, 0, i, env)
    shapley_cart_vel += shapley.shapley_value(policy, nc.pred, shapley.global_sverl_value_function, G, 1, i, env)
    shapley_pole_angle += shapley.shapley_value(policy, nc.pred, shapley.global_sverl_value_function, G, 2, i, env)
    shapley_pole_vel += shapley.shapley_value(policy, nc.pred, shapley.global_sverl_value_function, G, 3, i, env)
shapley_cart_pos /= NUM_ROUNDS
shapley_cart_vel /= NUM_ROUNDS
shapley_pole_angle /= NUM_ROUNDS
shapley_pole_vel /= NUM_ROUNDS
value_empty_set /= NUM_ROUNDS

print("Shapley value of Cart Position: ", shapley_cart_pos)
print("Shapley value of Cart Velocity: ", shapley_cart_vel)
print("Shapley value of Pole Angle: ", shapley_pole_angle)
print("Shapley value of Pole Angular Velocity: ", shapley_pole_vel)
print("Value of empty set: ", value_empty_set)



NUM_ROUNDS = 10
shapley_pos = 0
shapley_vel = 0
value_empty_set = 0
G= [[0,2], [1,3]] #The groups of features. In this case, we have 4 features, and each feature is its own group.
print("Calculating Shapley values based on NC...")
for i in trange(NUM_ROUNDS): 
    value_empty_set += shapley.global_sverl_value_function(policy, i, nc.pred, np.zeros(4), env)
    shapley_pos += shapley.shapley_value(policy, nc.pred, shapley.global_sverl_value_function, G, 0, i, env)
    shapley_vel += shapley.shapley_value(policy, nc.pred, shapley.global_sverl_value_function, G, 1, i, env)
shapley_pos /= NUM_ROUNDS
shapley_vel /= NUM_ROUNDS
value_empty_set /= NUM_ROUNDS

print("Shapley value of cart and angular position: ", shapley_pos)
print("Shapley value of cart and angular velocity: ", shapley_vel)
print("Value of empty set: ", value_empty_set)
