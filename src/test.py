import numpy as np
import gymnasium as gym 
import pickle 
from os.path import join
import time

env = gym.make("CartPole-v1", render_mode = "human")

policy = pickle.load(open(join("models", "cartpole_policy.pkl"), "rb"))

starting_state = np.array([1.5,0,0,0], dtype = np.int32)

state = env.reset()[0]

env.unwrapped.state = starting_state 


action = policy(state)
state, r, terminated, truncated, _ = env.step(action)
time.sleep(2)

state = starting_state


truncated = False 
terminated = False
reward = 0
while not (truncated or terminated): 
    action = policy(state)
    state, r, terminated, truncated, _ = env.step(action)
    reward += r

print("Reward: ", reward)