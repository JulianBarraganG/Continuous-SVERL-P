import gymnasium as gym  # Defines RL environments
from os.path import join, exists
import numpy as np

from sverl.imputation_utils import load_random_sampler, load_neural_conditioner, load_vaeac, get_policy_and_trajectory, StateFeatureDataset
from sverl.cartpole_agent import PolicyCartpole, train_cartpole_agent
from vaeac.train_utils import TrainingArgs

from sverl.shapley_utils import get_imputed_characteristic_dict, shapley_value, global_sverl_value_function
from sverl.sverl_utils import report_sverl_p



# Define the groups for group Shapley values
state_feature_names = ["Cart Position", "Cart Velocity", "Pole Angle", "Pole Angular Velocity"]
G = [[0, 2], [1, 3]]  # Grouped positional features and grouped velocity features

env = gym.make('CartPole-v1')

# CartPole one hot max sizes are all 0-s,
# since each feature is continous (i.e. real)
cp_one_hot_max_sizes = [0, 0, 0, 0]

action_space_dimension = env.action_space.n - 1 # This is the dimension, not the size 
state_space_dimension = env.observation_space.shape[0]

policy = PolicyCartpole(state_space_dimension, action_space_dimension)
model_filepath = join("models", "cartpole_policy.pkl")
trajectory_filename = "cartpole_trajectory"
rs_filepath = join("imputation_models", "cartpole_rs.pkl")
nc_filepath = join("imputation_models", "cartpole_nc.pkl")
vaeac_filepath = join("imputation_models", "cartpole_vaeac.pkl")
rs_characteristic_dict_filepath = join("characteristic_dicts", "cartpole_rs_characteristic_dict.pkl")
nc_characteristic_dict_filepath = join("characteristic_dicts", "cartpole_nc_characteristic_dict.pkl")
vaeac_characteristic_dict_filepath = join("characteristic_dicts", "cartpole_vaeac_characteristic_dict.pkl")
# Check if the model and trajectory files exist, otherwise train and save them
feature_imputation_model_missing = not(exists(rs_filepath) 
                                       and exists(nc_filepath) 
                                       and exists(vaeac_filepath))


# We assume trajectory file is csv
policy, trajectory = get_policy_and_trajectory(policy,
                                              env,
                                              model_filepath,
                                              trajectory_filename,
                                              train_cartpole_agent,
                                              gen_and_save_trajectory=feature_imputation_model_missing)

if feature_imputation_model_missing:
    batch_size = 32
    dataset = StateFeatureDataset(trajectory, batch_size=batch_size, shuffle=True)
    dataloader = dataset.dataloader
    input_dim = state_space_dimension  # size of the data
    # TODO: Investigate appropriate latent_dim in both cart pole lunar lander
    latent_dim = 64  # Size of the latent space

    nc = load_neural_conditioner(nc_filepath, input_dim=input_dim, latent_dim=latent_dim, dataloader=dataloader)
    rs = load_random_sampler(rs_filepath, trajectory=trajectory)
    vaeac = load_vaeac(vaeac_filepath, data=trajectory, args=TrainingArgs(), one_hot_max_sizes=cp_one_hot_max_sizes)
else: 
    nc = load_neural_conditioner(nc_filepath)
    rs = load_random_sampler(rs_filepath)
    vaeac = load_vaeac(vaeac_filepath)

# Move trained model to CPU
vaeac.cpu()

#The i is the seed. This is the only way I know how to set the starting position 

print("Calculating Shapley values based on RandomSampler...")

rs_char_dict = get_imputed_characteristic_dict(rs_characteristic_dict_filepath, env, policy, rs.pred, 10, global_sverl_value_function)
rs_shapley_values = np.zeros(len(state_feature_names)) 

for i in range(len(rs_shapley_values)): 
    rs_shapley_values[i] = shapley_value(i, rs_char_dict)  # Calculate Shapley value for each feature
report_sverl_p(rs_shapley_values, state_feature_names)

print("\nCalculating Shapley values based on NeuralConditioner...")
nc_char_dict = get_imputed_characteristic_dict(nc_characteristic_dict_filepath, env, policy, nc.pred, 10, global_sverl_value_function)
nc_shapley_values = np.zeros(len(state_feature_names)) 

for i in range(len(nc_shapley_values)): 
    nc_shapley_values[i] = shapley_value(i, nc_char_dict)  # Calculate Shapley value for each feature
report_sverl_p(nc_shapley_values, state_feature_names)


print("\nCalculating Shapley values based on VAEAC...")
vaeac_char_dict = get_imputed_characteristic_dict(vaeac_characteristic_dict_filepath, env, policy, vaeac.generate_probable_sample, 10, global_sverl_value_function)
vaeac_shapley_values = np.zeros(len(state_feature_names)) 

for i in range(len(vaeac_shapley_values)): 
    vaeac_shapley_values[i] = shapley_value(i, vaeac_char_dict)  # Calculate Shapley value for each feature
report_sverl_p(vaeac_shapley_values, state_feature_names)

