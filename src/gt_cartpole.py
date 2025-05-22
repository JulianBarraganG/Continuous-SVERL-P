from gymnasium import make
import numpy as np
from os.path import join

from sverl.cartpole_agent import PolicyCartpole, train_cartpole_agent
from sverl.sverl_utils import report_sverl_p
from sverl.shapley_utils import get_gt_characteristic_dict, shapley_value

# Instantiating variables
env = make("CartPole-v1")
savepath = join("characteristic_dicts", "gt_cartpole_characteristic_dict.pkl")
model_filepath = join("models", "cartpole_policy.pkl")
state_feature_names = ["Cart Position", "Cart Velocity", "Pole Angle", "Pole Angular Velocity"]
state_space_dim = env.observation_space.shape[0] # State space dimension
shapley_values = np.zeros(state_space_dim)  # Initialize Shapley values for each feature
empty_set_mask = np.array([0,0,0,0]).tobytes()
num_eval_eps = 10
num_train_eps = 1

# Get the ground truth characteristic dictionary
characteristic_dict = get_gt_characteristic_dict(savepath, env, PolicyCartpole, train_cartpole_agent, num_eval_eps, num_train_eps, model_filepath)

# Calculate the Shapley values for each feature
for i in range(state_space_dim):
    shapley_values[i] = shapley_value(i, characteristic_dict)  # Calculate Shapley value for each feature
empty_set_val = characteristic_dict[empty_set_mask]  # Get the value of the empty set

# Report the SVERL-P values
report_sverl_p(shapley_values, state_feature_names, row_name="GT_CP", data_file_prefix="cartpole")

