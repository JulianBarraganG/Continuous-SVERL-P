import gymnasium as gym  # Defines RL environments
from os.path import join, exists

from sverl.sverl_utils import (
        StateFeatureDataset,
        get_agent_and_trajectory, 
        load_neural_conditioner, 
        load_random_sampler, 
        load_vaeac, 
        get_sverl_p,
        report_sverl_p
        )
from sverl.cartpole_agent import PolicyCartpole, train_cartpole_agent
from vaeac.train_utils import TrainingArgs


# Define the groups for group Shapley values
state_feature_names = ["Cart Position", "Cart Velocity", "Pole Angle", "Pole Angular Velocity"]
pos_v_vel_G = [[0, 2], [1, 3]]  # Grouped positional features and grouped velocity features

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

# Move trained model to CPU
vaeac.cpu()

#The i is the seed. This is the only way I know how to set the starting position 

print("Calculating Shapley values based on RandomSampler...")
rs_shapley_values, rs_value_empty_set = get_sverl_p(policy, env, rs.pred)
report_sverl_p(rs_shapley_values, rs_value_empty_set, state_feature_names)

print("\nCalculating Shapley values based on NeuralConditioner...")
nc_shapley_values, nc_value_empty_set = get_sverl_p(policy, env, nc.pred)
report_sverl_p(nc_shapley_values, nc_value_empty_set, state_feature_names)

print("\nCalculating Shapley values based on VAEAC...")
vaeac_shapley_values, vaeac_value_empty_set = get_sverl_p(policy, env, vaeac.generate_probable_sample)
report_sverl_p(vaeac_shapley_values, vaeac_value_empty_set, state_feature_names)
