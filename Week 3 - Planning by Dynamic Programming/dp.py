import numpy as np

def one_step_lookahead(environment, state, V, discount_factor):
	"""
	Helper function to calculate the value function.
	
	PARAMETERS
	----------
	
	environment: Initialized OpenAI environment object.
	state: Agent's state to consider.
	V: The value to use as an estimator. Vector of length nS.
	discount_factor: Relative weightage of future rewards in an MDP.
	
	"""
	# Create a vector of dimensionality same as the number of actions
	action_values = np.zeros(environment.nA)
	
	for action in range(environment.nA):
		
		for probability, next_state, reward, terminated in environment.P[state][action]:
			action_values[action] += probability * (reward + discount_factor * V[next_state])
		
	return action_values


def policy_evaluation(policy, environment, discount_factor=1.0, theta=1e-9, max_iterations=1e9):
	"""
	Evaluate a policy given a deterministic environment.
	
	PARAMETERS
	----------
	
	policy: Matrix of size nS x nA. Each cell represents the probability of taking an
			action in a particular state.
	environment: Initialized OpenAI environment object.
	discount_factor: Relative weightage of future rewards in an MDP.
	theta: Convergence factor. If the change in value function for all the states is
		   below theta, we are done.
	max_iterations: Max number of iterations to prevent Infinite loops.
					
	RETURNS
	-------
	
	policy: The optimal policy.
	V: The optimal value estimate.
	"""
	
	# To record the number of iterations after which the evaluation converged
	evaluation_iterations = 1
	
	# Initialize the value function vector
	V = np.zeros(environment.nS)
	
	for i in range(int(max_iterations)):
		
		# For early stopping
		delta = 0
		
		for state in range(environment.nS):
			
			# Store the state's new value for this iteration
			v = 0
			
			# Check for all possible actions
			for action, action_probability in enumerate(policy[state]):
				
				# Iterate over all possible next states
				for state_probability, next_state, reward, terminated in environment.P[state][action]:
					
					# Calculate the expected value
					v += action_probability * state_probability * (reward + discount_factor * V[next_state])
				
			# Maintain the maximum change of value for each state
			delta = max(delta, abs(V[state] - v))
			
			# Update the state value
			V[state] = v
		
		# Update the number of iterations
		evaluation_iterations += 1
			
		# Early stopping
		if(delta < theta):
			print('Policy evaluated in %d iterations' % evaluation_iterations)
			return V


def policy_iteration(environment, discount_factor=1.0, max_iterations=1e9):
	"""
	Policy Iteration algorithm to solve a Markov Decision Process (MDP).
	
	Idea: Begin with a random policy function, use the Bellman expectation equation
	to calculate the value function corresponding to this policy and update the policy
	greedily using this value function at each iteration, until convergence.
	
	PARAMETERS
	----------
	
	environment: Initialized OpenAI environment object.
	discount_factor: Relative weightage of future rewards in an MDP.
	max_iterations: Max number of iterations to prevent Infinite loops.
					
	RETURNS
	-------
	
	policy: The optimal policy.
	V: The optimal value estimate.
	"""
	
	# Initialize the policy with a uniform distribution over the actions for each state
	policy = np.ones((environment.nS, environment.nA)) / environment.nA
	
	# Store the number of policies evaluated
	evaluated_policies = 1
	
	for i in range(int(max_iterations)):
		
		# For Early Stopping
		stable_policy = True
		
		# Evaluate the current policy
		V = policy_evaluation(policy, environment, discount_factor=discount_factor)
				
		for state in range(environment.nS):
			
			# Get the get action so far
			current_action = np.argmax(policy[state])
			
			# Perform the one-step lookahead to get the action values for the state
			action_values = one_step_lookahead(environment, state, V, discount_factor)
			
			# Get the best action
			best_action = np.argmax(action_values)
			
			# If the best action for the state changes, the policy is not yet stable
			if(current_action != best_action):
				stable_policy = False
				
			# Update the policy for the state
			policy[state] = np.eye(environment.nA)[best_action]
		
		# Increment the number of policies evaluated
		evaluated_policies += 1
		
		# Early stopping
		if(stable_policy):
			print('Evaluated %d policies.' % evaluated_policies)
			return policy, V


def value_iteration(environment, discount_factor=1.0, theta=1e-9, max_iterations=1e9):
	"""
	Value Iteration algorithm to solve a Markov Decision Process (MDP).
	
	Idea: Begin with a random value function, use the Bellman optimality equation
	to obtain a better value function at each iteration, until convergence. 
	Use this value function, to calculate the optimal policy.
	
	PARAMETERS
	----------
	
	environment: Initialized OpenAI environment object.
	discount_factor: Relative weightage of future rewards in an MDP.
	theta: Convergence factor. If the change in value function for all the states is
		   below theta, we are done.
	max_iterations: Max number of iterations to prevent Infinite loops.
					
	RETURNS
	-------
	
	policy: The optimal policy.
	V: The optimal value estimate.
	"""
	
	# Create a vector of dimensionality as the number of states
	V = np.zeros(environment.nS) 
	
	for i in range(int(max_iterations)):
		
		# early stopping condition
		delta = 0
		
		for state in range(environment.nS):
			
			# Perform the one-step lookahead to get the action values for the state
			action_values = one_step_lookahead(environment, state, V, discount_factor)
			
			# Get the best action value
			best_action_value = np.max(action_values)
			
			# Compute the maximum change in the value for each state
			delta = max(delta, abs(V[state] - best_action_value))
			
			# Update the best value for the state
			V[state] = best_action_value
			
		# Early stopping condition
		if(delta < theta):
			print('Value iteration converged at iteration #%d' % i)
			break
			
			
	# Find the optimal policy corresponding to the optimal value function
	policy = np.zeros((environment.nS, environment.nA))
	
	for state in range(environment.nS):
		
		action_values = one_step_lookahead(environment, state, V, discount_factor)
		
		# Choose the best action
		best_action = np.argmax(action_values)
		
		policy[state][best_action] = 1.0
	
	return policy, V
	
