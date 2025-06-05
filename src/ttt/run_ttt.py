from ttt_env import TTT
import numpy as np
import utils
from ttt_agent import Agent
import NeuralConditioner 
import shapley
from tqdm import trange
import PiSampler 

env = TTT()



agent = Agent()
print("Training agent...")


#want to report how good the policy is
print("Evaluating policy...")
no_evaluation_episodes = 100
reward = utils.evaluate_policy(no_evaluation_episodes, env, agent.choose_action) #Evaluating the policy
print("Average reward when running ", no_evaluation_episodes, " episodes: ", np.mean(reward)) #Printing the average reward
print("Standard deviation when running ", no_evaluation_episodes, " episodes:: ", np.std(reward)) #Printing the standard deviation of the reward


#EVERYTHING BELOW SHOULD HOPEFULLY BE GENERAL AT LEAST FOR ALL GYMNASIUM ENVS
print("Generating trajectories...")
trajectories_unflattened = utils.get_trajectory(agent.choose_action, env, time_horizon = 10**2) #Running the agent for 20 times, and storing the results
#Also storing the trajectories, which is used to train the Neural Conditioner

trajectories = []

for i in range(len(trajectories_unflattened)): 
    trajectories.append(trajectories_unflattened[i].flatten())

rs = PiSampler.PiSampler(np.array(trajectories))

#The i is the seed. This is the only way I know how to set the starting position 
#We are doing 100 different seeds, and averaging the results.
NUM_ROUNDS = 10

shapley_values = np.zeros(9)
initial_state = [[0,0,0],[0,1,0],[2,0,2]]
G= [[0], [1], [2], [3], [4], [5], [6], [7], [8]] #The groups of features. In this case, we have 4 features, and each feature is its own group.
print("Calculating Shapley values based on NC...")
for i in trange(NUM_ROUNDS): 
    for square in range(len(shapley_values)):
        np.random.seed(i)
        shapley_values[square] += shapley.shapley_value(agent.choose_action, rs.pred, shapley.local_sverl_value_function, G, square, initial_state, env)
  
shapley_values/NUM_ROUNDS

for square in range(len(shapley_values)): 
    print("Shapley value of square ", square, ": ", shapley_values[square])
