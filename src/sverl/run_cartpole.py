import gymnasium as gym  # Defines RL environments
import numpy as np  # For numerical operations
import shapley
import utils
import NeuralConditioner 
import agent
from tqdm import tqdm, trange  
from torch.utils.data import Dataset, DataLoader

env = gym.make('CartPole-v1')
state_space_dimension = env.observation_space.shape[0]


action_space_dimension = env.action_space.n - 1

policy = agent.Policy_cartpole(state_space_dimension, action_space_dimension)
print("Training agent...")
policy = agent.train_cartpole_agent(policy, env) #Here we train the agent, and report the evaluation steps

#want to report how good the policy is
print("Evaluating policy...")
no_evaluation_episodes = 100
reward = utils.evaluate_policy(no_evaluation_episodes, env, policy) #Evaluating the policy
print("Average reward when running ", no_evaluation_episodes, " episodes: ", np.mean(reward)) #Printing the average reward
print("Standard deviation when running ", no_evaluation_episodes, " episodes:: ", np.std(reward)) #Printing the standard deviation of the reward


#EVERYTHING BELOW SHOULD HOPEFULLY BE GENERAL AT LEAST FOR ALL GYMNASIUM ENVS
print("Generating trajectories...")
trajectories = utils.get_trajectory(policy, env, time_horizon = 10**4) #Running the agent for 20 times, and storing the results
#Also storing the trajectories, which is used to train the Neural Conditioner


dataset = utils.StateFeatureDataset(trajectories)

batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
input_dim = 4  # size of the data

latent_dim = 64  # Size of the latent space
nc = NeuralConditioner.NC(input_dim, latent_dim)
discriminator = NeuralConditioner.Discriminator(input_dim)
print("Training Neural Conditioner...")
NeuralConditioner.train_nc(nc, discriminator, dataloader, epochs=10)

#The i is the seed. This is the only way I know how to set the starting position 
#We are doing 100 different seeds, and averaging the results.
NUM_ROUNDS = 10
shapley_cart_pos = 0
shapley_cart_vel = 0
shapley_pole_angle = 0
shapley_pole_vel = 0
G= [[0], [1], [2], [3]] #The groups of features. In this case, we have 4 features, and each feature is its own group.
print("Calculating Shapley values...")
for i in trange(NUM_ROUNDS): 
    shapley_cart_pos += shapley.shapley_value(policy, nc, shapley.global_sverl_value_function, G, 0, i, env)
    shapley_cart_vel += shapley.shapley_value(policy, nc, shapley.global_sverl_value_function, G, 1, i, env)
    shapley_pole_angle += shapley.shapley_value(policy, nc, shapley.global_sverl_value_function, G, 2, i, env)
    shapley_pole_vel += shapley.shapley_value(policy, nc, shapley.global_sverl_value_function, G, 3, i, env)
shapley_cart_pos /= NUM_ROUNDS
shapley_cart_vel /= NUM_ROUNDS
shapley_pole_angle /= NUM_ROUNDS
shapley_pole_vel /= NUM_ROUNDS
print("Shapley value of Cart Position: ", shapley_cart_pos)
print("Shapley value of Cart Velocity: ", shapley_cart_vel)
print("Shapley value of Pole Angle: ", shapley_pole_angle)
print("Shapley value of Pole Angular Velocity: ", shapley_pole_vel)
