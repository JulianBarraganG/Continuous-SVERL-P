from tqdm import tqdm, trange  # Progress bar

import numpy as np

class Softmax_policy:
    def __init__(self, no_actions, no_features):
        """
        Initialize softmax policy for discrete actions
        :param no_actions: number of actions
        :param no_features: dimensionality of feature vector representing a state
        """        
        self.no_actions = no_actions
        self.no_features = no_features

        # Initialize policy parameters to zero
        self.theta = np.zeros([no_actions, no_features])
        
    def pi(self, s):
        """
        Compute action probabilities in a given state
        :param s: state feature vector
        :return: an array of action probabilities
        """
        # Compute action preferences for the given feature vector
        preferences = self.theta.dot(s)
        # Convert overflows to underflows
        preferences = preferences - preferences.max()
        # Convert the preferences into probabilities
        exp_prefs = np.exp(preferences)
        return exp_prefs / np.sum(exp_prefs)
    
    def inc(self, delta):
        """
        Change the parameters by addition, e.g. for initialization or parameter updates 
        :param delta: values to be added to parameters
        """
        self.theta += delta

    def sample_action(self, s):
        """
        Sample an action in a given state
        :param s: state feature vector
        :return: action
        """
        return np.random.choice(self.no_actions, p=self.pi(s))

    def gradient_log_pi(self, s, a):
        """
        Computes the gradient of the logarithm of the policy
        :param s: state feature vector
        :param a: action
        :return: gradient of the logarithm of the policy
        """
        d = np.zeros([self.no_actions, self.no_features])
        sum_term = np.sum([np.exp(np.dot(theta, s)) for theta in self.theta])
        for i in range(self.no_actions):
            numerator = np.exp(np.dot(self.theta[i], s))
            scalar_term = numerator/sum_term
            d[i] = (i == a)*s - scalar_term*s
           
        return d

    def gradient_log_pi_test(self, s, a, eps=0.1):
        """
        Numerically approximates the gradient of the logarithm of the policy
        :param s: state feature vector
        :param a: action
        :return: approximate gradient of the logarithm of the policy
        """
        theta_correct = np.copy(self.theta)
        log_pi = np.log(self.pi(s)[a])
        d = np.zeros([self.no_actions, self.no_features])
        for i in range(self.no_actions):
            for j in range(self.no_features):
                self.theta[i,j] += eps
                log_pi_eps = np.log(self.pi(s)[a])
                d[i,j] = (log_pi_eps - log_pi) / eps
                self.theta = np.copy(theta_correct)
        return d
  

def train(pi, env, no_episodes = 200 , alpha = 0.0005): 

    
    total_reward_list = []  # Returns for the individual episodes

    # Do the learning
    for e in trange(no_episodes):  #  Loop over episodes
        R = []  # Store rewards r_1, ..., r_T
        S = []  # Store actions a_0, ..., a_{T-1}
        A = []  # Store states s_0, ..., s_{T-1}
        state = env.reset()[0]  # Environment starts in a random state, cart and pole are moving
        while True:  # Environment sets "done" to true after 200 steps 
            S.append(state)
            
            action = pi.sample_action(state)  # Take an action following pi
            A.append(action)
            
            state, reward, terminated, truncated, _ = env.step(action)  # Observe reward and new state
            R.append(reward)
                    
            if terminated or truncated:  # Failed or succeeded?
                break
                
        R = np.array(R)
        total_reward_list.append((e, R.sum()))
        
        for t in range(R.size):
            R_t = R[t:].sum()  # Accumulated future reward
            Delta = alpha * R_t * pi.gradient_log_pi(S[t], A[t])  # REINFORCE update
            pi.inc(Delta)  # Apply update

    return pi



        