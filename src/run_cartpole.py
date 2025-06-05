import gymnasium as gym  # Defines RL environments
from os.path import exists
import numpy as np
from datetime import datetime
from gt_cartpole import get_gt_cartpole
from vaeac.train_utils import TrainingArgs

from sverl.UnifSampler import UnifSampler
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

# CartPole one hot max sizes are all 0-s,
# since each feature is continous (i.e. real)
policy = PolicyCartpole(state_space_dimension, action_space_dimension)



############################################# EXPERIMENT PREP #############################################
##### Experiment number
exp_num = 4
print(f"Running experiment number {exp_num}...")

### Get GT models and Shapley values
gt_shap = get_gt_cartpole(num_eval_eps=EVAL_ROUNDS, num_models=NUM_GT_MODELS)

# Get the respective NN size dict and latend dimension
cp_latent_dim = CP_LATENT_DIM[exp_num - 1]  # Latent dimension for VAEAC
cp_vaeac_nn_size_dict = {}  # Neural network size for VAEAC

for key in CP_VAEAC_NN_SIZE_DICT.keys():
    cp_vaeac_nn_size_dict[key] = CP_VAEAC_NN_SIZE_DICT[key][exp_num - 1]

###################################### TRAIN MODELS AND IMPUTERS ##########################################
# Check if the model and trajectory files exist, otherwise train and save them
feature_imputation_model_missing = not(exists(PI_SMP_FILEPATH) 
                                       and exists(NC_FILEPATH) 
                                       and exists(VAEAC_FILEPATH))

# We assume trajectory file is csv
policy, trajectory = get_policy_and_trajectory(policy,
                                              env,
                                              MODEL_FILEPATH,
                                              TRAJECTORY_FILENAME,
                                              train_cartpole_agent,
                                              gen_and_save_trajectory=feature_imputation_model_missing, 
                                              no_evaluation_episodes= EVAL_ROUNDS, 
                                              no_states_in_trajectories=TRAJECTORY_SIZE)

if feature_imputation_model_missing:
    dataset = StateFeatureDataset(trajectory, batch_size=BATCH_SIZE, shuffle=True)
    dataloader = dataset.dataloader
    input_dim = state_space_dimension  # size of the data

    nc = load_neural_conditioner(NC_FILEPATH, input_dim=input_dim, latent_dim=cp_latent_dim, dataloader=dataloader)
    pi_smp = load_random_sampler(PI_SMP_FILEPATH, trajectory=trajectory)
    vaeac = load_vaeac(VAEAC_FILEPATH, data=trajectory, args=TrainingArgs(), 
                       one_hot_max_sizes=CP_ONE_HOT_MAX_SIZES, nn_size_dict=cp_vaeac_nn_size_dict)
else: 
    nc = load_neural_conditioner(NC_FILEPATH)
    pi_smp = load_random_sampler(PI_SMP_FILEPATH)
    vaeac = load_vaeac(VAEAC_FILEPATH)
unif = UnifSampler(CP_RANGES)

# Move trained model to CPU
vaeac.cpu()

########################## CALC AND REPORT SHAPLEY VALUES ##########################

### Create a time based identity for storing data
dt = datetime.now()
id = "_exp_" + str(exp_num)  # Append experiment number to id
id += "_" + dt.strftime("%y%m%d%H%M") # formatted to "YYMMDDHHMM" as a len 10 str of digits

print("Calculating Shapley values based on Ground Truth models...")
report_sverl_p(gt_shap, CP_STATE_FEATURE_NAMES, row_name="GT_CP", data_file_name="cartpole" + id)

### Instantiate shapley value arrays and variables
nc_shapley_values = np.zeros(state_space_dimension) 
vaeac_shapley_values = np.zeros(state_space_dimension) 
pi_smp_shapley_values = np.zeros(state_space_dimension) 
unif_shapley_values = np.zeros(state_space_dimension)


print("Calculating Shapley values based on PiSampler...")
pi_smp_char_dict = get_imputed_characteristic_dict(PI_SMP_CHARACTERISITIC_DICT_FILEPATH,
                                               env,
                                               policy,
                                               pi_smp.pred,
                                               EVAL_ROUNDS, 
                                               global_sverl_value_function)

# Shapley values for Random Sampler
for i in range(state_space_dimension):
    pi_smp_shapley_values[i] = shapley_value(i, pi_smp_char_dict)  # Calculate Shapley value for each feature
report_sverl_p(pi_smp_shapley_values, CP_STATE_FEATURE_NAMES, row_name="PI_SMP", data_file_name="cartpole" + id)

print("Calculating Shapley values based on UnifSampler...")
unif_char_dict = get_imputed_characteristic_dict(UNIF_CHARACTERISITIC_DICT_FILEPATH,
                                               env,
                                               policy,
                                               unif.pred,
                                               EVAL_ROUNDS, 
                                               global_sverl_value_function)

# Shapley values for Off-Manifold Random Sampler
for i in range(state_space_dimension):
    unif_shapley_values[i] = shapley_value(i, unif_char_dict)  # Calculate Shapley value for each feature
report_sverl_p(unif_shapley_values, CP_STATE_FEATURE_NAMES, row_name="UNIF", data_file_name="cartpole" + id)

print("\nCalculating Shapley values based on NeuralConditioner...")
nc_char_dict = get_imputed_characteristic_dict(NC_CHARACTERISITIC_DICT_FILEPATH,
                                               env,
                                               policy,
                                               nc.pred,
                                               EVAL_ROUNDS,
                                               global_sverl_value_function)

# Shapley values for Neural Conditioner
for i in range(state_space_dimension):
    nc_shapley_values[i] = shapley_value(i, nc_char_dict)  # Calculate Shapley value for each feature
report_sverl_p(nc_shapley_values, CP_STATE_FEATURE_NAMES, row_name="NC", data_file_name="cartpole" + id)

print("\nCalculating Shapley values based on VAEAC...")
vaeac_char_dict = get_imputed_characteristic_dict(VAEAC_CHARACTERISITIC_DICT_FILEPATH,
                                                  env,
                                                  policy,
                                                  vaeac.generate_probable_sample,
                                                  EVAL_ROUNDS,
                                                  global_sverl_value_function)


# Shapley values for VAEAC
for i in range(state_space_dimension): 
    vaeac_shapley_values[i] = shapley_value(i, vaeac_char_dict)  # Calculate Shapley value for each feature
report_sverl_p(vaeac_shapley_values, CP_STATE_FEATURE_NAMES, row_name="VAEAC", data_file_name="cartpole" + id)

# Plot experiment results
plot_suffix = (str(exp_num) +
               "LD_" + str(cp_latent_dim) + "_W_" 
               + str(cp_vaeac_nn_size_dict["width"]) +
               "_D_" + str(cp_vaeac_nn_size_dict["depth"]))

# Save the experiment results in a plot (yay)
plot_data_from_id("cartpole" + id, "CP_" + plot_suffix)

