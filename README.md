Environment object

- state
- reward function
- reset
- step

step function should take an action and return the next state, the reward and done (done would mean that inventory is zero)

DQN Agent
- state_size
- action_size
- memory
- gamma
- epsilons
- learning_rate
- model 
- build model
- remember
- act
- replay


Parameters:
  
Paper's parameters
- Inventory: 20 lots (lot = 100 shares)
- Time constraint to execute: 1 hour
- Time Periods (within 1 execution): 5 (every 1hr/5 = 12 mins, fit, predict and change action accordingly)
- Time ticks: seconds (every second perform an action according to equally splitting the action derived from the last time period)
- Replay memory: 10,000
- Gamma: 0.99
- Exact day times used: 11-12pm, 12-1pm, 1-2pm
- Neural Network: 
    - 6 Layers of 20 nodes
    - Relu activation function (linear for last layer)
    - RMSprop Optimizer
- Neural Network Training
    - e-greedy: with Binomial ( q_(T_k), 1/(T_N-T_k) ) for non-optimal action.
        - q_(T_k) = remaining inventory
        - T_N - T_k = remaining time to execute
    - epsilons: not specified
    - batch size: 32