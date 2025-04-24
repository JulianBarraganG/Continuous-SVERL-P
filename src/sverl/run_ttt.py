from ttt_env import TTT
import numpy as np
import utils
from ttt_agent import Agent

env = TTT()



agent = Agent()
print("Training agent...")

agent.train(env)

#want to report how good the policy is
print("Evaluating policy...")
no_evaluation_episodes = 100
reward = utils.evaluate_policy(no_evaluation_episodes, env, agent.choose_action) #Evaluating the policy
print("Average reward when running ", no_evaluation_episodes, " episodes: ", np.mean(reward)) #Printing the average reward
print("Standard deviation when running ", no_evaluation_episodes, " episodes:: ", np.std(reward)) #Printing the standard deviation of the reward




