import gymnasium as gym  # Defines RL environments
import numpy as np  # For numerical operations
import shapley
import utils
import NeuralConditioner 
import lunar_agent
from tqdm import tqdm, trange  
from torch.utils.data import Dataset, DataLoader


env = gym.make('LunarLander-v3', render_mode="rgb_array") 
action_space_dimension = 4
state_space_dimension = 8

print("Training agent...")
policy = lunar_agent.train(env, train_episodes=2) #Here we train the agent and get a policy

#Want to report how good the policy is
print("Evaluating policy...")
no_evaluation_episodes = 100
reward = utils.evaluate_policy(no_evaluation_episodes, env,  policy.predict) #Evaluating the policy
print("Average reward when running ", no_evaluation_episodes, " episodes: ", np.mean(reward)) #Printing the average reward
print("Standard deviation when running ", no_evaluation_episodes, " episodes:: ", np.std(reward)) #Printing the standard deviation of the reward

#EVERYTHING BELOW SHOULD HOPEFULLY BE GENERAL AT LEAST FOR ALL GYMNASIUM ENVS
print("Generating trajectories...")
trajectories = utils.get_trajectory(policy.predict, env, time_horizon = 10**3) #Running the agent for a time-horizon of 10**4, and storing the trajectories, 
#which is used to train the neural conditioner


dataset = utils.StateFeatureDataset(trajectories)

batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
input_dim = 8  # size of the data. the state-space

latent_dim = 64  # Size of the latent space
nc = NeuralConditioner.NC(input_dim, latent_dim)
discriminator = NeuralConditioner.Discriminator(input_dim)

print("Training Neural Conditioner...")
NeuralConditioner.train_nc(nc, discriminator, dataloader, epochs=5)

#The i is the seed. This is the only way I know how to set the starting position 
#We are doing 10 different seeds, and averaging the results.
NUM_ROUNDS = 10
shapley_pos = 0
shapley_vel = 0
shapley_angle = 0
shapley_angle_vel = 0
shapley_leg_bools = 0
value_empty_set = 0
G = [[0, 1], [2, 3], [4], [5], [6, 7]] #The groups of features. In this case, we have 8 features, and each feature is its own group.
print("Calculating Shapley values...")
for i in trange(NUM_ROUNDS): 
    value_empty_set += shapley.global_sverl_value_function(policy.predict, i, nc.pred, np.zeros(8), env)
    shapley_pos += shapley.shapley_value(policy.predict, nc.pred, shapley.global_sverl_value_function, G, 0, i, env)
    shapley_vel += shapley.shapley_value(policy.predict, nc.pred, shapley.global_sverl_value_function, G, 1, i, env)
    shapley_angle += shapley.shapley_value(policy.predict, nc.pred, shapley.global_sverl_value_function, G, 2, i, env)
    shapley_angle_vel += shapley.shapley_value(policy.predict, nc.pred, shapley.global_sverl_value_function, G, 3, i, env)
    shapley_leg_bools += shapley.shapley_value(policy.predict, nc.pred, shapley.global_sverl_value_function, G, 4, i, env)
shapley_pos /= NUM_ROUNDS
shapley_vel /= NUM_ROUNDS
shapley_angle /= NUM_ROUNDS
shapley_angle_vel /= NUM_ROUNDS
shapley_leg_bools /= NUM_ROUNDS
value_empty_set /= NUM_ROUNDS
print("Value of empty set: ", value_empty_set)
print("Shapley value of he coordinates of the lander: ", shapley_pos)
print("Shapley value of linear velocities: ", shapley_vel)
print("Shapley value of Angle: ", shapley_angle)
print("Shapley value of Angular Velocity: ", shapley_angle_vel)
print("Shapley value of leg booleans: ", shapley_leg_bools)   

