# Week 4: Model-Free Prediction

This week covers the model free prediction algorithms like Monte-Carl and TD Learning. The code demonstrates these algorithms in different environments.

## Theory
- The algorithms we studied in Week 3 required that the model knows the entire dynamics of the environment and can compute what 
the outcome of its actions will be before actually performing the action. This is the reason it was called *Planning*.

- Harder RL problem is when the model doesn't know anything about the environment and instead depends on its iteractions with the environment by taking actions and observing the environment, until it learns good policy to do so. There are two approachess for learning the optimal policy without knowing the transition probabilities and the reward function.

- **Model-based**: Learn a model from its observation and use that model to *plan* (via Policy / Value Iteration) its solution.
- **Model-Free**: Rely on trial-and-error experience for action selection. Here, we miss the transition probabilities as well
                  as the reward function. There are again 2 approaches for this:
    
    - **Passive Learning**: Here, the model always performs a particular action in a particular state and simply learns the                                 utility of being in that state. This it the case of *Model Free Prediction*.
    
    - **Active Learning**: Estimate the optimal policy by moving in the environment. This is 
                           the case of *Model Free Control* ([Week 5](https://github.com/dalmia/David-Silver-Reinforcement-learning/tree/master/Week%205)).
                           
Characteristics of Model free prediction algorithms:

- Policy already given
- Learn directly from episodes of experience. 

### Algorithms

- Monte-Carlo Learning: 
  - Applicable to episodic MDPs (those which terminate)
  - Idea - Use the empirical mean (instead of the expectation) of sample returns across many episodes as the value for a particular state. This works because of the law of large numbers.
  - 2 Methods: First-Visit (count only the first time a state is visited in an episode) and Every-Visit (multiple visits possible for each state in an episode).
  - Mean is calculated incrementally with a weight added to decrease the significance of older values.
 
- Temporal Difference Learning
  
  
