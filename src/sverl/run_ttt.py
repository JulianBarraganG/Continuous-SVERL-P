from TTTenv import TTT
import numpy as np
import shapley
import utils
import NeuralConditioner 
import RandomSampler
from tqdm import tqdm, trange  
import ttt_agent

env = TTT()



agent = ttt_agent.Agent(env.observation_space.shape[0], env.num_actions, epsilon=0.05, gamma=1, alpha=0.2)
print("Training agent...")

ttt_agent.train(agent, env, 1e6)

##This is to demonstrate that the agent is not learning.

state, _ = env.reset()
print(state)
state,reward, done, _, _ = env.step(agent.choose_action(state))
print(state, reward, done)
state,reward, done, _, _ = env.step(agent.choose_action(state))
print(state, reward, done)
state,reward, done, _, _ = env.step(agent.choose_action(state))
print(state, reward, done)
state,reward, done, _, _ = env.step(agent.choose_action(state))
print(state, reward, done)
state,reward, done, _, _ = env.step(agent.choose_action(state))
print(state, reward, done)

