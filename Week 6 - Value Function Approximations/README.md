# Week 6: Value Function Approximations

This week covers the value function approximations which make RL scalable and were the key components behind the famous Deep Q-Networks (DQN).
The code demonstrates these algorithms in different environments.

## Summary

- The methods that have been described earlier are not practical for the kind of problems that RL are applied to because of the sheer number of states and corresponding actions possible. For large-scale problems, we use **Function Approximators**. This week covers Value Function Approximations (VFA), while the next week looks at Policy Based Approximations.

- There are two kinds of approaches to function approximations:
  - **Incremental Method**: Update value function online upon seeing new data.
  - **Batch Methods**: Update value function in a batch.

- The idea is to estimate the value function for the entire state (and action) space using only a set of parameters *W*.
  The advantages are two fold:
  - It saves memory.
  - It allows generalization to unseen states.

- We have many different function approximators (FAs) like Decision Tree, Neural Networks, etc. Here, we stick to only those FAs which are differentiable - *Linear combination of features* and *Neural Networks*. The challenge compared to Supervised Learning (SL) is that the data is non-iid.

- Use Gradient descent for updating the weights according to the cost function. 

- A feature vector `x(S)` with `n` values - each value describes one specific thing about the state. 

- Linear VFA: Take a linear combination of the features and use the mean square error between the estimated value and the true value as the loss function. The benefit is that the cost function is now quadratic. 

- Earlier, we did a table lookup, which is just a special case of linear VFA, with the value in row `i` of `x(S)` being the indicator function, `S = Si`. 

- We don't have  the true value. Replace the true value with MC / TD(0) / TD(λ) estimate. In the case of TD(λ), the eligibility trace is on the parameters and not the states. 

- Control with VFA:
  - Approximate Policy Evaluation (on the action value function)
  - ϵ-greedy policy improvement.
  
  Action-Value Function Approximation works the same as VFA, by using `q` function instead of `v` and the feature vector is given for a state-action pair, `x(S, A)`.
  
- MC almost always reaches a bad result because of too much variance and the need for much more data for learning. This means, bootstrapping works better with TD(λ) performing better than TD(0). However, TD doesn't guarantee stability as it uses a biased estimate of the value function and the fact that TD doesn't follow *any* objective function. Various improvements like Gradient TD get rid of the stability problem, by taking the true gradient of the projected Bellman error. Control is problematic for action value approximation, with the final policy chattering around the optimal policy in many cases.

- The problem with incremental methods is that it is not sample efficient, i.e. it throws away experience after one use. **Batch methods** find the best fitting value to the entire batch. Use Least Squares fit to minimize the error over the entire dataset `D`.

- An easy way to find the least squares solution is to use **Experience Relay (ER)** the entire experience is stored and the objective is to accurately fit to data randomly sampled from that experience using Stochastic Gradient Descent (SGD). The global optimum reached is the true Least Squares solution. ER along with fixed Q-targets played an important role in maintaining the stability of Deep Q-Networks [DQN] (FA -> Big NN). Fixed Q-targets refers to freezing the model at an earlier stage and using that to specify the Q-learning targets (making them fixed).

- In case of linear FA, we can jump directly to the optimal solution using the fact that at the optimal point, the update for the parameters would be 0.

## Results

- Q-Learning with Value Function Approximation:
  ![value][results/value.png]
  ![reward][results/reward.png]
 
## References
- https://github.com/dennybritz/reinforcement-learning/tree/master/DQN
- https://github.com/dennybritz/reinforcement-learning/tree/master/FA
