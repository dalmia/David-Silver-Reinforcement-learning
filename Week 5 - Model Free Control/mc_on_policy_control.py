
# coding: utf-8

# In[1]:


import gym
import numpy as np
import matplotlib
import sys
from collections import defaultdict


if "../" not in sys.path:
    sys.path.append("../") 
    
from lib.envs.blackjack import BlackjackEnv
from lib import plotting

matplotlib.style.use('ggplot')


# In[2]:


env = BlackjackEnv()


# In[34]:


def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.

    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.

    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    
    """
    
    def policy_fn(observation):
        
        # Get the action values corresponding to the observation
        action_values = Q[observation]
        
        # Get the greedy action
        greedy_action = np.argmax(action_values)
        
        # Choose a random action with probability epsilon / nA
        probabilities = np.ones(nA) * epsilon / nA
        
        # Choose the greedy action with probability (1 - epsilon)
        probabilities[greedy_action] += (1 - epsilon)
        
        return probabilities
        
    return policy_fn    


# In[30]:


def mc_control_epsilon_greedy(env, num_episodes, discount_factor=1.0, epsilon=0.1):
    """
    First-Visit Monte Carlo On-Policy Control using Epsilon-Greedy policies.
    Finds an optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI gym environment.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
    
    Returns:
        A tuple (Q, policy).
        Q is a dictionary mapping state -> action values.
        policy is a function that takes an observation as an argument and returns
        action probabilities
    """
    
    # Number of actions
    nA = env.action_space.n
    
    # Action Value function to be returned
    Q = defaultdict(lambda: np.zeros(nA))
    
    # The optimal policy to be returned
    policy = make_epsilon_greedy_policy(Q, epsilon, nA)
    
     # store the number of times each state is visited 
    returns_num = defaultdict(float) 
        
    for episode in range(1, num_episodes + 1):
        
        # GLIE schedule
        epsilon /= episode
        
        # store the eligibility trace corresponding to each state-action pair for each episode
        eligibility_traces = defaultdict(float)
        
        # store the reward corresponding to each state-action for each episode
        episode_rewards = defaultdict(float)
        
        terminated = False
        state = env.reset()
        
        # termination condition
        while not terminated:
            
            # sample the action from the epsilon greedy policy
            action = np.random.choice(nA, p=policy(state))
            
            # update the eligibility trace for the state-action pairs already visited in the episode 
            for (_state, _action) in eligibility_traces:
                
                eligibility_traces[(_state, _action)] *= discount_factor
            
            # add a new state-action pair to the dictionary if it's not been visited before
            if (state, action) not in eligibility_traces:
                
                eligibility_traces[(state, action)] = 1.0
                returns_num[(state, action)] += 1
            
            # perform the action in the environment
            next_state, reward, terminated, _ = env.step(action)
            
            # update the reward for each state-action pair
            for (_state, _action) in eligibility_traces:
                
                episode_rewards[(_state, _action)] += eligibility_traces[(_state, _action)] * reward
            
            # update the current state
            state = next_state
        
        # update the action value function using incremental mean method
        for (state, action) in episode_rewards:
            Q[state][action] += (episode_rewards[(state, action)] - Q[state][action]) / returns_num[(state, action)]
        
        # Policy Improvement
        policy = make_epsilon_greedy_policy(Q, epsilon, nA)
        
    return Q, policy


# In[35]:


def find_optimal_policy(num_episodes):
    Q, policy = mc_control_epsilon_greedy(env, num_episodes=num_episodes, epsilon=0.1)
    
    # For plotting: Create value function from action-value function
    # by picking the best action at each state
    V = defaultdict(float)
    for state, actions in Q.items():
        action_value = np.max(actions)
        V[state] = action_value
    plotting.plot_value_function(V, title="Optimal Value Function ({} episodes)".format(num_episodes))


# In[36]:


find_optimal_policy(500000)

