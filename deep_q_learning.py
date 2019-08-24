import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import pandas as pd

class State(object):
  def __init__(self, time, inventory):
    self.time = time # period (integer)
    self.inventory = inventory # number of shares yet to be executed

class Env(object):
  def __init__(self, state, prices):
    self.state = state
    self.prices = prices
    
  def reset(self, number_of_shares):
    self.state = State(0, number_of_shares)

  def step(self, action):
    '''
    - action is an integer number of shares bought in one period
    - step function should take an action and return the next state, the reward and done (done would mean that inventory is zero)
    - also returns the immediate reward and the boolean done
    '''
    new_inventory = self.state.inventory - action
    new_time = self.state.time + 1
    reward = reward(self.state.inventory, action)
    self.state = State(new_time, new_inventory)

    return (self.state, reward, is_done)

  def is_done():
    return self.state.inventory == 0

  def get_price(self, time):
    return prices[time]

  def reward(self, remaining_inventory, action, a=0.01):
    return remaining_inventory*(get_price(time+1) -  get_price(time)) - a*(action**2)

EPISODES = 1000

def run():

  df_raw = pd.read_csv('./Data_Sets/APPL10minTickData.csv', header=0)
  prices = np.array(df_raw['close'].to_numpy)

  initial_state = State(inventory=100, time=0)
  env = Env(state=initial_state, prices=prices)

  state_size = 2
  action_size = env.state.inventory
  agent = DQNAgent(state_size, action_size)

  done = False
  batch_size = 32

  # for e in range(EPISODES):
  
class DQNAgent:
    def __init__(self, state_size, action_size, time_to_execute):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99 # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.time_to_execute = 42 # multiples of 10 minutes: 7 hours
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        # TODO: update the architecutre --> <=6 layers
        # Geometric pyramid rule
        # Start with uniform for e-greedy, then go Binomial
        # normalise (time)
        # https://www.quora.com/What-are-some-rules-of-thumb-for-training-neural-networks
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
      # TODO: action size (if put later)
      if np.random.rand() <= self.epsilon:
          return random.randrange(self.action_size)
      act_values = self.model.predict(state)
      return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

if __name__ == "__main__":
  run()