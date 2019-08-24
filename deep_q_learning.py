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

  initial_state = State(inventory=1000, time=0)
  env = Env(state=initial_state, prices=prices)

  inventory = 1000
  time  = 100 # multiples of 10 minutes
  state_space = np.array([time, inventory])
  action_space = np.array(list(range(0, len(inventory))))
  state_size = len(state_space)
  action_size = len(action_space)
  agent = DQNAgent(state_size, action_size)
  # agent.load("./save/cartpole-dqn.h5")
  done = False
  batch_size = 32
  price = np.array([]) #insert full price time series

  print("training is starting")
  for e in range(EPISODES):
      state = state_space
      state = np.reshape(state, [1, state_size])  # what's the point??? TODO
      for time in range(500):
          action = agent.act(state)
          next_state = (inventory - action)
          price_over_t = price[(time-1):(time+noOfSteps)] #price vector should include the following times [t-1, t, t+1, ..., t+noOfSteps]
          # if time%%12 == 0
          #   action = agent.act(state)
              reward = agent.reward(next_state, action, price_over_t,noOfSteps)
          # else:
          #   reward = agent.reward(next_state, action/12, price_over_t,noOfSteps)
          reward = agent.reward(next_state, action, price_over_t,noOfSteps)
          next_state = np.reshape(next_state, [1, state_size])  # ??? TODO
          done = True if inventory == 0 else False
          agent.remember(state, action, reward, next_state, done)
          state = next_state
          if done:
              print("episode: {}/{}, score: {}, e: {:.2}"
                    .format(e, EPISODES, time, agent.epsilon))
              break
          if len(agent.memory) > batch_size:
              agent.replay(batch_size)
      # if e % 10 == 0:
      #     agent.save("./save/cartpole-dqn.h5")
      print("episode is finished")

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95 # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
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