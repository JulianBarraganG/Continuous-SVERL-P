from os.path import join

# EXPIRMENTAL PARAMETERS
EVAL_ROUNDS = 10**3  # Number of evaluation rounds for each imputation method
NUM_RT_MODELS = 16  # Number of competing RT models per coalition
TRAJECTORY_SIZE = 2*10**5  # Number of sampled trajectories for pi^\star (best policy)

# TRAINING PARAMETERS
CP_LATENT_DIM = [2, 4, 8, 16, 32]  # Size of the latent space for CP NC 
BATCH_SIZE = 32

# SEED FOR REPRODUCIBILITY
RESET_SEED = 42

# CARTPOLE ENVIRONMENT PARAMETERS
CP_STATE_FEATURE_NAMES = ["Cart Position", "Cart Velocity", "Pole Angle", "Pole Angular Velocity"]
CP_RANGES = [(-2.4, 2.4), ("ninf", "inf"), (-0.2095, 0.2095), ("ninf", "inf")]
POS_VEL_G = [[0, 2], [1, 3]]  # Grouped positional features and grouped velocity features
CP_ONE_HOT_MAX_SIZES = [0, 0, 0, 0]
CP_VAEAC_NN_SIZE_DICT = {
    "width": [32, 32, 64, 64, 128],  # Width of the hidden layers
    "depth": [4, 6, 8, 10, 12],  # Depth of the network (number of hidden layers)
    "latent_dim": 2*CP_LATENT_DIM,  # VAEAC latent dim is 2x CP_LATENT_DIM
}

# LUNAR LANDER ENVIRONMENT PARAMETERS
G = [[0, 1], [2, 3], [4], [5], [6, 7]]
LL_G_FEATURE_NAMES= ["Coordinates", "Linear Velocities", "Angle",
                       "Angular Velocity", "Touching Ground"]

LL_STATE_FEATURE_NAMES = ["X Coordinate", "Y Coordinate", "X Linear Velocity",
                            "Y Linear Velocity", "Angle", "Angular Velocity",
                            "Left Leg", "Right Leg"]

# CARTPOLE FILEPATHS FOR MODELS, DATA AND CHARACTERISTIC DICTS
MODEL_FILEPATH = join("models", "cartpole_policy.pkl")
TRAJECTORY_FILENAME = "cartpole_trajectory"
PI_SMP_FILEPATH = join("imputation_models", "cartpole_rs.pkl")
NC_FILEPATH = join("imputation_models", "cartpole_nc.pkl")
VAEAC_FILEPATH = join("imputation_models", "cartpole_vaeac.pkl")
PI_SMP_CHARACTERISITIC_DICT_FILEPATH = join("characteristic_dicts", "cartpole_rs_characteristic_dict.pkl")
UNIF_CHARACTERISITIC_DICT_FILEPATH = join("characteristic_dicts", "cartpole_ors_characteristic_dict.pkl")
NC_CHARACTERISITIC_DICT_FILEPATH = join("characteristic_dicts", "cartpole_nc_characteristic_dict.pkl")
VAEAC_CHARACTERISITIC_DICT_FILEPATH = join("characteristic_dicts", "cartpole_vaeac_characteristic_dict.pkl")
RT_CHARACTERISTIC_DICT_FILEPATH = join("characteristic_dicts", "rt_cartpole_characteristic_dict.pkl")

