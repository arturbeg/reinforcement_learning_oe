import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import pandas as pd
import collections

EPISODES = 100
INITIAL_INVENTORY = 100
TEN_MINUTES_IN_ONE_DAY = 42
A = 0.01

# TODO: put loggers for future debugging
# TODO: make sure by the end of the day no shares are left in the inventory

def run():
    df_raw = pd.read_csv('./Data_Sets/APPL10minTickData.csv', header=0)
    prices = np.array(df_raw['close'].to_numpy())

    initial_state = State(inventory=INITIAL_INVENTORY, time=0)
    env = Env(state=initial_state, prices=prices)

    state_size = 2
    action_size = INITIAL_INVENTORY
    agent = DQNAgent(state_size, action_size)

    done = False
    batch_size = 32

    PandL_agent_array = np.array([])
    PandL_TWAP_array = np.array([])

    for e in range(EPISODES):

        if df_raw.shape[0] - env.state.time <= TEN_MINUTES_IN_ONE_DAY + 1:
            state = env.full_reset()
        else:
            state = env.reset_inventory()

        state = np.reshape(state.state_as_list(), [1, state_size])

        for time in range(TEN_MINUTES_IN_ONE_DAY):
            action = agent.act(state=state, time=time)
            next_state, reward, done = env.step(action)
            next_state = next_state.state_as_list()
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            PandL_agent_array = np.append(PandL_agent_array, env.PandL_agent(action))
            print("Inventory is: " + str(env.state.inventory))
            print("Time is: " + str(env.state.time))
            # TODO: make sure done is implemented correctly
            if done:
                print("DONE")
                print("episode: {}/{}, P&L: {}, time: {}, e: {:.2}"
                      .format(e, EPISODES, sum(PandL_agent_array), time, agent.epsilon))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        #TWAP loop
        time_counter = 0
        twap_actions = np.array([])
        for time in range(TEN_MINUTES_IN_ONE_DAY):
            twap_actions= np.append(twap_actions,INITIAL_INVENTORY/TEN_MINUTES_IN_ONE_DAY)
            PandL_TWAP_array = np.append(PandL_TWAP_array, env.PandL_TWAP(INITIAL_INVENTORY, TEN_MINUTES_IN_ONE_DAY, time_counter))
            time_counter += 1

    total_PandL_agent = sum(PandL_agent_array)
    total_PandL_TWAP = sum(PandL_TWAP_array)
    PandL_vs_TWAP = ((total_PandL_agent-total_PandL_TWAP)/total_PandL_TWAP)*100
    print("PandL_vs_TWAP is {}".format(PandL_vs_TWAP) )

class State(object):
  def __init__(self, time, inventory):
    self.time = time # period (integer)
    self.inventory = inventory # number of shares yet to be executed

  def state_as_list(self):
    # return the state in the correct format
    return [self.time, self.inventory]

class Env(object):
  def __init__(self, state, prices):
    self.state = state
    self.prices = prices

  def full_reset(self):
    self.state = State(0, INITIAL_INVENTORY)
    return self.state

  def reset_inventory(self):
    self.state.inventory = INITIAL_INVENTORY
    return self.state

  def step(self, action):
    '''
    - action is an integer number of shares bought in one period
    - step function should take an action and return the next state, the reward and done (done would mean that inventory is zero)
    - also returns the immediate reward and the boolean done
    '''
    reward = self.reward(self.state.inventory, action)
    self.state.time = self.state.time + 1
    self.state.inventory = self.state.inventory - action
    return (self.state, reward, self.is_done())

  def is_done(self):
      #TODO: or if the time constraint finished. Add in the action function that it should execute all of the remaining inventory at the last time step
    return self.state.inventory <= 0

  def get_price(self, time):
    return self.prices[time]

  def reward(self, remaining_inventory, action, a=A):
    return remaining_inventory*(self.get_price(self.state.time+1) -  self.get_price(self.state.time)) - a*(action**2)

  def PandL_agent(self, action, a = A):
      return action*self.get_price(self.state.time) - a*(action**2)

  def PandL_TWAP(self, initial_inventory, time_constraint, time, a = A):
        action = initial_inventory/time_constraint
        return action * self.get_price(time) - a*(action**2)



class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99 # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        # self.time_to_execute = 42 # multiples of 10 minutes: 7 hours
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

    def act(self, state, time):
      if time == TEN_MINUTES_IN_ONE_DAY:
        return state[0][1]
      # here the state is already a numpy array
      # TODO: action size (if put later)
      if np.random.rand() <= self.epsilon:
          action = random.randrange(self.action_size)
          if action > state[0][1]:
            return state[0][1]
          else:
            return action
      act_values = self.model.predict(state)
      # put docstring specifying what act returns
      action = np.argmax(act_values[0])
      # TODO: make sure this works properly)
      if action > state[0][1]:
        return state[0][1]
      else:
        return action

    def replay(self, batch_size):
      '''
        replay uses the memory values to train the network
      '''
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