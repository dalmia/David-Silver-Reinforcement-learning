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

# Functions to find the optimal policy
solvers = [
	('Policy Iteration', policy_iteration),
	('Value Iteration', value_iteration)
]

for iteration_name, iteration_func in solvers:

	# Load the frozen lake Environment
	environment = gym.make('FrozenLake8x8-0')

	# Find the optimal policy using the corresponding solver function
	policy, V = iteration_func(environment.env)


