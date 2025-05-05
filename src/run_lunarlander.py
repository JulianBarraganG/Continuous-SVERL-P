import gymnasium as gym  # Defines RL environments
from os.path import join, exists

from sverl.sverl_utils import (
        StateFeatureDataset,
        get_policy_and_trajectory, 
        load_neural_conditioner, 
        load_random_sampler, 
        load_vaeac, 
        get_sverl_p,
        report_sverl_p
        )
from sverl.lunar_agent import QNetwork, train_Qnetwork
from vaeac.train_utils import TrainingArgs

G = [[0, 1], [2, 3], [4], [5], [6, 7]]
group_feature_names = ["Coordinates", "Linear Velocities", "Angle" , "Angular Velocity", "Touching Ground"]
env = gym.make('LunarLander-v3', render_mode="rgb_array") 

# One hot max sizes for the LunarLander environment
# 0-s are for continuous features (real valued) and 2-s are for binary inputs.
ll_one_hot_max_sizes = [0, 0, 0, 0, 0, 0, 2, 2]

action_space_dimension = env.action_space.n
state_space_dimension = env.observation_space.shape[0]

untrained_policy = QNetwork(state_space_dimension, action_space_dimension, hidden_size=64)
model_filepath = join("models", "lunar_policy.pkl")
trajectory_filename = "lunar_trajectory"
rs_filepath = join("imputation_models", "lunar_rs.pkl")
nc_filepath = join("imputation_models", "lunar_nc.pkl")
vaeac_filepath = join("imputation_models", "lunar_vaeac.pkl")

# Check if the model and trajectory files exist, otherwise train and save them
feature_imputation_model_missing = not(exists(rs_filepath) 
                                       and exists(nc_filepath) 
                                       and exists(vaeac_filepath))

# We assume trajectory file is csv
policy, trajectory = get_policy_and_trajectory(untrained_policy,
                                              env,
                                              model_filepath,
                                              trajectory_filename,
                                              train_Qnetwork,
                                              gen_and_save_trajectory=feature_imputation_model_missing)

if feature_imputation_model_missing:
    batch_size = 32
    dataset = StateFeatureDataset(trajectory, batch_size=batch_size, shuffle=True)
    dataloader = dataset.dataloader
    input_dim = state_space_dimension # size of the data
    latent_dim = 64  # Size of the latent space

    nc = load_neural_conditioner(nc_filepath, input_dim=input_dim, latent_dim=latent_dim, dataloader=dataloader)
    rs = load_random_sampler(rs_filepath, trajectory=trajectory)
    vaeac = load_vaeac(vaeac_filepath, data=trajectory, args=TrainingArgs(), one_hot_max_sizes=ll_one_hot_max_sizes)
else: 
    nc = load_neural_conditioner(nc_filepath)
    rs = load_random_sampler(rs_filepath)
    vaeac = load_vaeac(vaeac_filepath)

# Move trained model to CPU
vaeac.cpu()

# The i is the seed. This is the only way I know how to set the starting position 
#################### Print Shapley values for individual features ####################
print("Calculating Shapley values based on RandomSampler...")
rs_shapley_values, rs_value_empty_set = get_sverl_p(policy, env, rs.pred)
report_sverl_p(rs_shapley_values, rs_value_empty_set, group_feature_names)

print("\nCalculating Shapley values based on NeuralConditioner...")
nc_shapley_values, nc_value_empty_set = get_sverl_p(policy, env, nc.pred)
report_sverl_p(nc_shapley_values, nc_value_empty_set, group_feature_names)

print("\nCalculating Shapley values based on VAEAC...")
vaeac_shapley_values, vaeac_value_empty_set = get_sverl_p(policy, env, vaeac.generate_probable_sample)
report_sverl_p(vaeac_shapley_values, vaeac_value_empty_set, group_feature_names)

#################### Print shapley values for groups of features ####################
# print("\nCalculating Group Shapley values based on RandomSampler...")
# rs_group_shapley_values, rs_group_value_empty_set = get_sverl_p(policy, env, rs.pred, G=G)
# report_sverl_p(rs_group_shapley_values, rs_group_value_empty_set, group_feature_names, G=G)
# 
# print("\nCalculating Group Shapley values based on NeuralConditioner...")
# nc_group_shapley_values, nc_group_value_empty_set = get_sverl_p(policy, env, nc.pred, G=G)
# report_sverl_p(nc_group_shapley_values, nc_group_value_empty_set, group_feature_names, G=G)
# 
# print("\nCalculating Group Shapley values based on VAEAC...")
# vaeac_group_shapley_values, vaeac_group_value_empty_set = get_sverl_p(policy, env, vaeac.generate_probable_sample, G=G)
# report_sverl_p(vaeac_group_shapley_values, vaeac_group_value_empty_set, group_feature_names, G=G)
