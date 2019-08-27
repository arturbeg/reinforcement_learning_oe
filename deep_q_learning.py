import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import pandas as pd
from sklearn import preprocessing
import collections

def normalise(x):
    return (x-min(x))/(max(x)-min(x))

DF_RAW = pd.read_csv('./Data_Sets/APPL10minTickData.csv', header=0)

EPISODES = 100
INITIAL_INVENTORY = 101
INITIAL_INVENTORY_SCALED = 1
TIME_CONSTRAINT_FOR_EXECUTION = 50
A = 0.01
TRAINING = True

# DATA PREP
EXECUTION_TIMES_SCALED = normalise(np.array(range(TIME_CONSTRAINT_FOR_EXECUTION)))
TIME_CONSTRAINT_FOR_EXECUTION_SCALED = max(EXECUTION_TIMES_SCALED)
TIME_POINTS_FOR_EXECUTION = len(EXECUTION_TIMES_SCALED)
TRAIN_OBSERVATIONS = int(0.7 * len(DF_RAW))
if TRAINING == True:
    DF = DF_RAW.iloc[:TRAIN_OBSERVATIONS, ]
    REAL_TIME = 0
    END_TIME = len(DF)
else:
    DF = DF_RAW.iloc[TRAIN_OBSERVATIONS:, ]
    REAL_TIME = TRAIN_OBSERVATIONS
    END_TIME = TRAIN_OBSERVATIONS + len(DF)

PRICES = np.array(DF['close'].to_numpy())

# TODO: put loggers for future debugging
def run():

    initial_state = State(inventory=INITIAL_INVENTORY_SCALED, time=0)
    env = Env(initial_state, PRICES, TIME_CONSTRAINT_FOR_EXECUTION_SCALED, TIME_POINTS_FOR_EXECUTION, REAL_TIME, END_TIME)

    state_size = 2
    action_size = INITIAL_INVENTORY
    agent = DQNAgent(state_size, action_size)
    twap = TWAP(INITIAL_INVENTORY_SCALED, TIME_POINTS_FOR_EXECUTION)

    done = False
    batch_size = 32

    PandL_agent_array = np.array([])
    PandL_TWAP_array = np.array([])
    PandL_vs_TWAP_array = ([])

    for e in range(EPISODES):
        state = env.reset_game()
        state = np.reshape(state.state_as_list(), [1, state_size])
        print("REAL TIME is: " + str(env.real_time))
        print("start episode:" + str(e))
        for time in EXECUTION_TIMES_SCALED:
            print("inventory is: " + str(env.state.inventory))
            print("time is: " + str(env.state.time))
            action = agent.act(state, time, env.time_constraint_for_execution)
            next_state, reward, done = env.step(action)
            next_state = next_state.state_as_list()
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            PandL_agent_array = np.append(PandL_agent_array, env.PandL(action))
            PandL_TWAP_array = np.append(PandL_TWAP_array, env.PandL(twap.act()))
            PandL_vs_TWAP_array = np.append(PandL_vs_TWAP_array, ((PandL_agent_array[env.real_time-1] - PandL_TWAP_array[env.real_time-1]) / PandL_TWAP_array[env.real_time-1]) * 100)

            if done:
                print("DONE")
                print("episode: {}/{}, P&L_vs_TWAP: {}%, time: {}, e: {:.2}"
                      .format(e, EPISODES, PandL_vs_TWAP_array[env.real_time-1], time, agent.epsilon))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

    total_PandL_agent = sum(PandL_agent_array)
    total_PandL_TWAP = sum(PandL_TWAP_array)
    PandL_vs_TWAP = ((total_PandL_agent-total_PandL_TWAP)/total_PandL_TWAP)*100
    agent.save('model_weights.h5', PandL_vs_TWAP_array,PandL_agent_array, PandL_TWAP_array)
    print("PandL_vs_TWAP is {}%".format(PandL_vs_TWAP) )

class State(object):
  """"
    time = game time from t = 0 to t = TIME_CONSTRAINT_FOR_EXECUTION_SCALED
  """""

  def __init__(self, time, inventory):
    self.time = time # period (integer)
    self.inventory = inventory # number of shares yet to be executed

  def state_as_list(self):
    # return the state in the correct format
    return [self.time, self.inventory]

class TWAP(object):
    def __init__(self, initial_inventory, time_points_for_execution):
        self.initial_inventory = initial_inventory
        self.time_points_for_execution = time_points_for_execution

    def act(self):
        action = self.initial_inventory / self.time_points_for_execution
        return action

class Env(object):
  def __init__(self, state, prices, time_constraint_for_execution, time_points_for_execution, real_time, end_time):
    self.state = state
    self.prices = prices
    self.time_constraint_for_execution = time_constraint_for_execution
    self.time_points_for_execution = time_points_for_execution
    self.real_time = real_time
    self.end_time = end_time

  def reset_game(self):
    self.state.inventory = INITIAL_INVENTORY_SCALED
    self.state.time = 0
    return self.state

  def step(self, action):
    '''
    - action is an integer number of shares bought in one period
    - step function should take an action and return the next state, the reward and done (done would mean that inventory is zero)
    - also returns the immediate reward and the boolean done
    '''
    reward = self.reward(self.state.inventory, action, self.real_time)
    self.state.time = round(self.state.time + self.time_constraint_for_execution/self.time_points_for_execution, 2)
    self.real_time = (self.real_time + 1) % self.end_time
    self.state.inventory = round(self.state.inventory - action,2)
    return (self.state, reward, self.is_done())

  def is_done(self):
    return self.state.time == self.time_constraint_for_execution  #to make sure our agent is comparable with TWAP time-wise

  def get_price(self):
    return self.prices[self.real_time]

  def reward(self, remaining_inventory, action, a=A):
    return remaining_inventory*(self.get_price() -  self.get_price()) - a*(action**2)

  def PandL(self, action, a = A):
      return action*self.get_price() - a*(action**2)

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

    def act(self, state, time, time_constraint_for_execution, egreedy = 'Binomial'):
      # TODO: make sure this works properly
      inventory = state[0][1]

      if time == time_constraint_for_execution:
        action = inventory
      elif inventory == 0:
        action = 0 #to make it's time consistent with TWAP for comparison purposes
      elif np.random.rand() <= self.epsilon:
          if egreedy == 'Binomial':
              n = inventory*INITIAL_INVENTORY
              p = (1/TIME_POINTS_FOR_EXECUTION)/(time_constraint_for_execution - time)
              action = np.random.binomial(n, p)
              action =  np.linspace(0,1,101)[action] #scale back action to a normalised action
          elif egreedy == 'Uniform':
              action = random.randrange(self.action_size)
      else:
          act_values = self.model.predict(state) # put docstring specifying what act returns
          action = np.argmax(act_values[0])

      if action > inventory:
          action = inventory
      print("action is: " + str(action))
      return round(action,2)

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
          action_range = np.linspace(0, 1, 101)
          print("Minibatch action is: " + str(action))
          action_index = [i for i in range(len(action_range)) if round(action_range[i].item(),2) == action][0]
          target_f[0][action_index] = target
          self.model.fit(state, target_f, epochs=1, verbose=0)
      if self.epsilon > self.epsilon_min:
          self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name, PandL_vs_TWAP, PandL_agent, PandL_TWAP):
        self.model.save_weights(name)
        state_time = np.array([item[0][0][0] for item in list(self.memory)])
        state_inventory = np.array([item[0][0][1] for item in list(self.memory)])
        actions = np.array([item[1] for item in list(self.memory)])
        rewards = np.array([item[2] for item in list(self.memory)])
        next_state_time = np.array([item[3][0][0] for item in list(self.memory)])
        next_state_inventory = np.array([item[3][0][1] for item in list(self.memory)])
        done = np.array([item[4] for item in list(self.memory)])
        memory_df = pd.DataFrame({'state_time': state_time, 'state_inventory': state_inventory, 'action': actions, 'reward': rewards,'next_state - Inventory': next_state_time, 'next_state - Time': next_state_inventory, 'done':done, 'PandL_vs_TWAP': PandL_vs_TWAP, 'PandL_agent': PandL_agent, 'PandL_TWAP': PandL_TWAP})
        print(memory_df)
        memory_df.to_csv('MemoryAndP&L.csv')



if __name__ == "__main__":
  run()
