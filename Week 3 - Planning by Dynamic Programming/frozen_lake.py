import gym
import numpy as np

from dp import policy_iteration, value_iteration

# Action mappings - Map actions to numbers
action_mappings = {
	0: '\u2191', # UP
	1: '\u2192', # RIGHT
	2: '\u2193', # DOWN
	3: '\u2190', # LEFT
}


def play_episodes(environment, n_episodes, policy):
	pass


# Number of episodes to play
n_episodes = 10000


def play_episodes(environment, n_episodes, policy):
	wins = 0
	total_reward = 0
	
	for episode in range(n_episodes):
		
		terminated = False
		state = environment.reset()
		
		while not terminated:
			
			# Choose the best action according to the policy
			action = np.argmax(policy[state])
			
			# Perform the action in the environment
			next_state, reward, terminated, info = environment.step(action)
			
			# Add the reward to the total reward count
			total_reward += reward
			
			# Update the current state
			state = next_state
			
			# Check if the episode is over and reward achieved
			if terminated and reward == 1.0:
				wins += 1
				
	average_reward = total_reward / n_episodes
	
	return wins, total_reward, average_reward
			
		
	
# Functions to find the optimal policy
solvers = [
	('Policy Iteration', policy_iteration),
	('Value Iteration', value_iteration)
]

	
for iteration_name, iteration_func in solvers:

	# Load the frozen lake Environment
	environment = gym.make('FrozenLake8x8-v0')
	print('Environment Created.')

	# Find the optimal policy using the corresponding solver function
	print('Running %s ...' % iteration_name)
	policy, V = iteration_func(environment.env)
	print('Done.')
	
	print('\n Final policy derived using {iteration_name}:')
	print(''.join([action_mappings[action] for action in np.argmax(policy, axis=1)]))
	
	# Use the learnt policy for playing a few episodes of the game
	wins, total_reward, average_reward = play_episodes(environment, n_episodes, policy)
	
	print('%s :: number of wins over %d episodes = %d' % (iteration_name, n_episodes, wins))
	print('%s :: average reward over %d episodes = %.2f \n' % (iteration_name, n_episodes, average_reward))
