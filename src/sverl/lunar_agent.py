from tqdm import tqdm, trange  # Progress bar
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

import numpy as np

class QNetwork(nn.Module):
    def __init__(self, state_size=8, action_size=4, hidden_size=10, bias=True):
        """
        Q-Network for reinforcement learning.
        
        Parameters
        -------------
        state_size: int 
            Size of the state space.
        action_size : int 
            Size of the action space.
        hidden_size : int 
            Number of neurons in the hidden layer.
        bias : bool 
            Whether to include bias in the linear layers.
        """
        super(QNetwork, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.fc1 = nn.Linear(state_size, hidden_size, bias)  
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias)  
        self.output_layer = nn.Linear(hidden_size + state_size, action_size, bias)

    def forward(self, x_input):
        """
        Forward pass through the network.
        
        Parameters
        -------------
            x_input : torch.Tensor 
                Input state tensor.
        Returns
        -------------
            torch.Tensor: 
                Q-values for each action.
        """
        x = F.tanh(self.fc1(x_input))
        x = F.tanh(self.fc2(x))
        x = torch.cat((x_input, x), dim=1)
        x = self.output_layer(x)
        return x
    def predict(self, x_input): 
        """
        Predict the action based on the input state.
        
        Parameters
        -------------
            x_input : numpy.ndarray 
                Input state array.
        Returns
        -------------
            int: 
                Predicted action (index of the action with the highest Q-value).
        """
        x_input = torch.from_numpy(x_input).unsqueeze(0)
        x = F.tanh(self.fc1(x_input))
        x = F.tanh(self.fc2(x))
        x = torch.cat((x_input, x), dim=1)
        x = self.output_layer(x)
        x = torch.argmax(x).item()
        return x

    

class Memory():
    def __init__(self, max_size = 1000):
        """
        Experience replay memory for storing past experiences.
        
        Parameters
        -------------
            max_size : int 
                Maximum size of the memory buffer.
        """
        self.buffer = deque(maxlen=max_size)
    
    def add(self, experience):
        """
        Add an experience to the memory buffer.
        
        Parameters
        -------------
            experience : tuple 
                Experience tuple (state, action, reward, next_state).
        """
        self.buffer.append(experience)
            
    def sample(self, batch_size):
        """
        Sample a batch of experiences from the memory buffer randomly 
        
        Parameters
        -------------
            batch_size : int
                Number of experiences to sample.
        Returns
        -------------
            list:
                List of sampled experiences."""
        idx = np.random.choice(np.arange(len(self.buffer)), 
                               size=batch_size, 
                               replace=False)
        return [self.buffer[ii] for ii in idx]
    

def train_Qnetwork(mainQN, 
                   env,
                   train_episodes = 400, 
                   gamma = 0.99, 
                   learning_rate = 0.001, 
                   tau = .01, 
                   explore_start = 1.0, 
                   explore_stop = 0.0001, 
                   decay_rate = 0.05, 
                   memory_size = 10000, 
                   batch_size = 128):
    """
    Train a Q-learning agent using experience replay and delayed target network soft updates.
    
    Parameters
    -------------
    env :  gym.Env
        Environment to train the agent on. We use it on the lunar lander 
    train_episodes : int
        Number of training episodes.
    hidden_size : int
        Number of neurons in the hidden layer of the Q-networks
    gamma : float
        Discount factor for future rewards.
    learning_rate : float
        Learning rate for the optimizer.
    tau : float
        Soft update parameter for the target network.
    explore_start : float
        Initial exploration probability.
    explore_stop : float
        Final exploration probability.
    decay_rate : float
        Decay rate for exploration probability.
    memory_size : int
        Maximum size of the experience replay memory.
    batch_size : int
        Size of the mini-batch for training.
    
    Returns
    -------------
        targetQN (QNetwork): 
            Target Q-network after training. This network is used for action selection, and should be 
            more stable than the main Q-network.
    """
    pretrain_length = batch_size
    targetQN = QNetwork(mainQN.state_size, mainQN.action_size, mainQN.hidden_size, mainQN.bias)
    state = env.reset()[0]
    action_size = env.action_space.n
    state_size = env.observation_space.shape[0]
    memory = Memory(max_size=memory_size)

    # Make a bunch of random actions and store the experiences
    for _ in range(pretrain_length):
        # Make a random action
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            # The simulation fails, so no next state
            next_state = np.zeros(state.shape)
            # Add experience to memory
            memory.add((state, action, reward, next_state))
            
            # Start new episode
            env.reset()
            # Take one random step to get the pole and cart moving
            state, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        else:
            # Add experience to memory
            memory.add((state, action, reward, next_state))
            state = next_state

    optimizer = torch.optim.AdamW(mainQN.parameters(), lr=learning_rate) # AdamW uses weight decay by default
    loss_fn = torch.nn.MSELoss()

    for ep in trange(train_episodes):
        total_reward = 0  # Return / accumulated rewards
        state = env.reset()[0]  # Reset and get initial state
        while True:
            # Explore or exploit
            explore_p = explore_stop + (explore_start - explore_stop)*np.exp(-decay_rate*ep) 
            if explore_p > np.random.rand():
                # Pick a random action
                action = env.action_space.sample()
            else:
                # Get action from Q-network
                state_tensor = torch.from_numpy(np.resize(state, (1, state_size)).astype(np.float32))
                Qs = mainQN(state_tensor)
                action = torch.argmax(Qs).item()

            # Take action, get new state and reward
            next_state, reward, terminated, truncated, _ = env.step(action)
        
            total_reward += reward  # Return / accumulated rewards
            
            if terminated or truncated:
                # Episode ends because of failure, so no next state
                next_state = np.zeros(state.shape)
                    
                # Add experience to memory
                memory.add((state, action, reward, next_state))
                break; # End of episode
            else:
                # Add experience to memory
                memory.add((state, action, reward, next_state))
                state = next_state
                
            # Sample mini-batch from memory
            batch = memory.sample(batch_size)
            next_states_np = np.array([each[3] for each in batch], dtype=np.float32)
            next_states = torch.as_tensor(next_states_np)  # as_tensor does not copy the data
            rewards     = torch.as_tensor(np.array([each[2] for each in batch], dtype=np.float32)) 
            states      = torch.as_tensor(np.array([each[0] for each in batch], dtype=np.float32))
            actions     = torch.as_tensor(np.array([each[1] for each in batch], dtype = np.int64))
                
            # Compute Q values for all actions in the new state       
            target_Qs = mainQN(next_states)
                
            # Set target_Qs to 0 for states where episode ended because of failure
            episode_ends = (next_states_np == np.zeros(states[0].shape)).all(axis=1)
            target_Qs[episode_ends] = torch.zeros(action_size)            
            
            # Compute targets
            with torch.no_grad():
                y = rewards + gamma * torch.max(target_Qs, dim=1).values       

            # Network learning starts here
            optimizer.zero_grad()
            
            # Compute the Q values of the actions taken        
            main_Qs = mainQN(states)  # Q values for all action in each state
            Q = torch.gather(main_Qs, np.int64(1), actions.unsqueeze(-1)).squeeze()  # Only the Q values for the actions taken
            
            # Gradient-based update
            loss = loss_fn(Q, y)
            loss.backward()
            optimizer.step()

            #Update target network

            sdTargetQN = targetQN.state_dict()
            sdMainQN = mainQN.state_dict() 

            for key in sdTargetQN:
                sdTargetQN[key] = tau * sdMainQN[key] + (1 - tau) * sdTargetQN[key]

            targetQN.load_state_dict(sdTargetQN)

    return targetQN    