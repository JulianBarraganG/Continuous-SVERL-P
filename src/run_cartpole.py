import gymnasium as gym  # Defines RL environments
from os.path import join, exists
import numpy as np
from datetime import datetime
from gt_cartpole import get_gt_cartpole
from vaeac.train_utils import TrainingArgs

from sverl.OffRandomSampler import OffRandomSampler
from sverl.imputation_utils import load_random_sampler, load_neural_conditioner, load_vaeac, get_policy_and_trajectory, StateFeatureDataset
from sverl.cartpole_agent import PolicyCartpole, train_cartpole_agent
from sverl.shapley_utils import get_imputed_characteristic_dict, shapley_value
from sverl.sverl_utils import report_sverl_p, global_sverl_value_function, local_sverl_value_function
from sverl.plotting import plot_data_from_id
from sverl.globalvars import *


########################################## VARIABLE DECLARATIONS ##########################################
#### Environment specific variables ####
# Define the groups for group Shapley values
env = gym.make('CartPole-v1')

action_space_dimension = env.action_space.n - 1 # This is the dimension, not the size 
state_space_dimension = env.observation_space.shape[0]

# Experiment variablse
eval_rounds = 10**3 # Number of evaluation rounds for each imputation method
num_gt_models = 16 # Number of competing GT models per coalition
trajectory_size = 10**5 # Number of sampled trajectories for pi^\star (best policy)

# CartPole one hot max sizes are all 0-s,
# since each feature is continous (i.e. real)
cp_one_hot_max_sizes = [0, 0, 0, 0]
policy = PolicyCartpole(state_space_dimension, action_space_dimension)

#### Filepaths for pkls and csv data ####
model_filepath = join("models", "cartpole_policy.pkl")
trajectory_filename = "cartpole_trajectory"
rs_filepath = join("imputation_models", "cartpole_rs.pkl")
nc_filepath = join("imputation_models", "cartpole_nc.pkl")
vaeac_filepath = join("imputation_models", "cartpole_vaeac.pkl")
rs_characteristic_dict_filepath = join("characteristic_dicts", "cartpole_rs_characteristic_dict.pkl")
ors_characteristic_dict_filepath = join("characteristic_dicts", "cartpole_ors_characteristic_dict.pkl")
nc_characteristic_dict_filepath = join("characteristic_dicts", "cartpole_nc_characteristic_dict.pkl")
vaeac_characteristic_dict_filepath = join("characteristic_dicts", "cartpole_vaeac_characteristic_dict.pkl")



########################################## TRAIN MODELS AND IMPUTERS ##########################################

### Get GT models and Shapley values
gt_shap = get_gt_cartpole(num_eval_eps=eval_rounds, num_models=num_gt_models)

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
                                              gen_and_save_trajectory=feature_imputation_model_missing, 
                                              no_evaluation_episodes= eval_rounds, 
                                              no_states_in_trajectories=trajectory_size)

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
ors = OffRandomSampler(CP_RANGES)

# Move trained model to CPU
vaeac.cpu()

########################## CALC AND REPORT SHAPLEY VALUES ##########################

### Create a time based identity for storing data
dt = datetime.now()
id = dt.strftime("%y%m%d%H%M") # formatted to "YYMMDDHHMM" as a len 10 str of digits

print("Calculating Shapley values based on Ground Truth models...")
report_sverl_p(gt_shap, CP_STATE_FEATURE_NAMES, row_name="GT_CP", data_file_name="cartpole" + id)

### Instantiate shapley value arrays and variables
nc_shapley_values = np.zeros(state_space_dimension) 
vaeac_shapley_values = np.zeros(state_space_dimension) 
rs_shapley_values = np.zeros(state_space_dimension) 
ors_shapley_values = np.zeros(state_space_dimension)

# For local
starting_state = np.array([1,0,0,0], dtype = np.int32)

print("Calculating Shapley values based on RandomSampler...")
rs_char_dict = get_imputed_characteristic_dict(rs_characteristic_dict_filepath, 
                                               env,
                                               policy,
                                               rs.pred,
                                               eval_rounds, 
                                               global_sverl_value_function)

# Shapley values for Random Sampler
for i in range(len(rs_shapley_values)): 
    rs_shapley_values[i] = shapley_value(i, rs_char_dict)  # Calculate Shapley value for each feature
report_sverl_p(rs_shapley_values, CP_STATE_FEATURE_NAMES, row_name="RS", data_file_name="cartpole" + id)

print("Calculating Shapley values based on OffRandomSampler...")
ors_char_dict = get_imputed_characteristic_dict(ors_characteristic_dict_filepath, 
                                               env,
                                               policy,
                                               ors.pred,
                                               eval_rounds, 
                                               global_sverl_value_function)

# Shapley values for Off-Manifoldl Random Sampler
for i in range(len(ors_shapley_values)): 
    ors_shapley_values[i] = shapley_value(i, ors_char_dict)  # Calculate Shapley value for each feature
report_sverl_p(ors_shapley_values, CP_STATE_FEATURE_NAMES, row_name="ORS", data_file_name="cartpole" + id)

print("\nCalculating Shapley values based on NeuralConditioner...")
nc_char_dict = get_imputed_characteristic_dict(nc_characteristic_dict_filepath,
                                               env,
                                               policy,
                                               nc.pred,
                                               eval_rounds,
                                               global_sverl_value_function)

# Shapley values for Neural Conditioner
for i in range(len(nc_shapley_values)): 
    nc_shapley_values[i] = shapley_value(i, nc_char_dict)  # Calculate Shapley value for each feature
report_sverl_p(nc_shapley_values, CP_STATE_FEATURE_NAMES, row_name="NC", data_file_name="cartpole" + id)

print("\nCalculating Shapley values based on VAEAC...")
vaeac_char_dict = get_imputed_characteristic_dict(vaeac_characteristic_dict_filepath,
                                                  env,
                                                  policy,
                                                  vaeac.generate_probable_sample,
                                                  eval_rounds,
                                                  global_sverl_value_function)


# Shapley values for VAEAC
for i in range(len(vaeac_shapley_values)): 
    vaeac_shapley_values[i] = shapley_value(i, vaeac_char_dict)  # Calculate Shapley value for each feature
report_sverl_p(vaeac_shapley_values, CP_STATE_FEATURE_NAMES, row_name="VAEAC", data_file_name="cartpole" + id)

# Plot experiment results
plot_data_from_id("cartpole" + id, "cartpole_results" + id)

