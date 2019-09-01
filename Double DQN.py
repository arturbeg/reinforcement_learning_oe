import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import pandas as pd
import collections
from keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
K.clear_session()


def normalise(x):
    return (x-min(x))/(max(x)-min(x))

#Code Controls
EPISODES = 500
MEMORY = 100000
INITIAL_INVENTORY = 21
INITIAL_INVENTORY_SCALED = 1
TIME_CONSTRAINT_FOR_EXECUTION = 11
A = 0.01

TRAINING = True
TRAIN_BOUNDARIES= False
LOAD_PRETRAINED_WEIGHTS = True

PANDL_REWARD = False
EGREEDY = "Binomial"
NN = "NN=6_NN"
OPTIMIZER = 'RMSprop'
FILENAME = "DDQN{}ep_{}_{}A={}Actions={}TimeConstr={}Opt={}REW=".format(EPISODES,NN, EGREEDY,A, INITIAL_INVENTORY, TIME_CONSTRAINT_FOR_EXECUTION, OPTIMIZER,str(PANDL_REWARD))
STATE_SIZE = 2
BATCH_SIZE = 64

# DATA PREP
DF_RAW = pd.read_csv('./Data_Sets/APPL10minTickData.csv', header=0)
EXECUTION_TIMES_SCALED = normalise(np.array(range(TIME_CONSTRAINT_FOR_EXECUTION)))
TIME_CONSTRAINT_FOR_EXECUTION_SCALED = max(EXECUTION_TIMES_SCALED)
TIME_POINTS_FOR_EXECUTION = len(EXECUTION_TIMES_SCALED)
TIME_UNIT = 1/(TIME_POINTS_FOR_EXECUTION-1)
TRAIN_OBSERVATIONS = int(0.7 * len(DF_RAW))
if TRAINING == True:
    DF = DF_RAW.iloc[:TRAIN_OBSERVATIONS, ]
    REAL_TIME = 0
    END_TIME = len(DF)
else:
    DF = DF_RAW.iloc[TRAIN_OBSERVATIONS:, ]
    REAL_TIME = TRAIN_OBSERVATIONS
    END_TIME = TRAIN_OBSERVATIONS + len(DF)
    EPISODES = 34

PRICES = np.array(DF['close'].to_numpy())
PRICES = normalise(PRICES)

def run():
    ''''
    Main Function
    Output: tables of the results and the weights of the trained agent
    '''''
    initial_state = State(inventory=INITIAL_INVENTORY_SCALED, time=0)
    env = Env(initial_state, PRICES, TIME_CONSTRAINT_FOR_EXECUTION_SCALED, TIME_POINTS_FOR_EXECUTION, REAL_TIME, END_TIME)

    state_size = STATE_SIZE
    action_size = INITIAL_INVENTORY
    agent = DQNAgent(state_size, action_size, TRAINING)
    if TRAINING != True:
        agent.load("{}_weights.h5".format(FILENAME))
    if (TRAINING == True and LOAD_PRETRAINED_WEIGHTS == True):
        agent.load("DDQN25ep_NN=6_NN_BinomialA=0.01Actions=21TimeConstr=11Opt=RMSpropREW=_weights.h5")
    twap = TWAP(INITIAL_INVENTORY_SCALED, TIME_POINTS_FOR_EXECUTION)

    done = False
    batch_size = BATCH_SIZE

    PandL_agent_array = np.array([])
    PandL_TWAP_array = np.array([])
    PandL_vs_TWAP_array = np.array([])
    rewards_array = np.array([])
    avg_rewards = np.array([])

    for e in range(EPISODES):
        state = env.reset_game()
        state = np.reshape(state.state_as_list(), [1, state_size])
        print("REAL TIME is: " + str(env.real_time))
        print("start episode:" + str(e))
        for time in EXECUTION_TIMES_SCALED:
            print("inventory is: " + str(env.state.inventory))
            print("time is: " + str(env.state.time))
            if TRAIN_BOUNDARIES == True:
                if time == 0:
                    index = np.random.binomial(1, 1 / 2)
                action = agent.act_boundary_conditions(state, time, env.time_constraint_for_execution, index)
            else:
                action = agent.act(state, time, env.time_constraint_for_execution)
            next_state, reward, done = env.step(action)
            next_state = next_state.state_as_list()
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            PandL_agent_array = np.append(PandL_agent_array, env.PandL(action))
            PandL_TWAP_array = np.append(PandL_TWAP_array, env.PandL(twap.act()))
            PandL_vs_TWAP_array = np.append(PandL_vs_TWAP_array, ((PandL_agent_array[env.real_time-1- REAL_TIME] - PandL_TWAP_array[env.real_time-1- REAL_TIME]) / PandL_TWAP_array[env.real_time-1- REAL_TIME]) * 100)
            rewards_array = np.append(rewards_array, reward)

            if done:
                agent.update_target_model()
                print("DONE")
                print("episode: {}/{}, P&L_vs_TWAP: {}%, time: {}, e: {:.2}".format(e, EPISODES, PandL_vs_TWAP_array[env.real_time-1-REAL_TIME], time, agent.epsilon))
                avg_rewards = np.append(avg_rewards, np.mean(rewards_array))
                rewards_array = np.array([])
                break
            if (len(agent.memory) > batch_size and TRAINING == True):
                agent.replay(batch_size)

    total_PandL_agent = sum(PandL_agent_array)
    total_PandL_TWAP = sum(PandL_TWAP_array)
    PandL_vs_TWAP = ((total_PandL_agent-total_PandL_TWAP)/total_PandL_TWAP)*100
    print("PandL_vs_TWAP is {}%".format(PandL_vs_TWAP) )

    avg_rewards_df = pd.DataFrame({'avg_rewards': avg_rewards})
    print("Avg_rewards are {}".format(avg_rewards))
    avg_rewards_df.to_csv('avg_rewards_DDQN{}Train={}NN={}REW=.csv'.format(EPISODES, TRAINING,NN, str(PANDL_REWARD)))

    agent.save("DDQN{}_weights.h5".format(FILENAME), PandL_vs_TWAP_array,PandL_agent_array, PandL_TWAP_array)
    plt.plot(avg_rewards)
    plt.show()


class State(object):


  def __init__(self, time, inventory):
    self.time = time # time period in an episode (integer)
    self.inventory = inventory # number of shares yet to be executed

  def state_as_list(self):
    # return the state in the correct format
    return [self.time, self.inventory]

class TWAP(object):
    ''''
    Class defining the TWAP strategy
    '''''

    def __init__(self, initial_inventory, time_points_for_execution):
        self.initial_inventory = initial_inventory
        self.time_points_for_execution = time_points_for_execution

    def act(self):
        action = self.initial_inventory / self.time_points_for_execution
        return action

class Env(object):
  ''''
  Class of the environment
  '''''
  def __init__(self, state, prices, time_constraint_for_execution, time_points_for_execution, real_time, end_time):
    self.state = state
    self.prices = prices
    self.time_constraint_for_execution = time_constraint_for_execution
    self.time_points_for_execution = time_points_for_execution #number of discrete time points where execution can happen within an episode
    self.real_time = real_time #time period along the whole of our data set (does not reset at each episode)
    self.end_time = end_time #final time of our dataset

  def reset_game(self):
    self.state.inventory = INITIAL_INVENTORY_SCALED
    self.state.time = 0
    return self.state

  def step(self, action):
    '''
    - step function should take an action and return the next state, the reward and done (done would mean that the episode has finished)
    - action is an integer number of shares bought in one period
    '''
    reward = self.reward(self.state.inventory, action)
    self.state.time = round(self.state.time + TIME_UNIT, 2)
    self.real_time = (self.real_time + 1) if (self.real_time +1) % self.end_time != 0 else REAL_TIME
    self.state.inventory = round(self.state.inventory - action,2)
    return (self.state, reward, self.is_done())

  def is_done(self):
    '''
      Determines when an episode ends. Returns boolean
    '''
    return self.state.time == self.time_constraint_for_execution

  def get_price(self):
    return self.prices[self.real_time - REAL_TIME]

  def reward(self, remaining_inventory, action):
      if PANDL_REWARD == True:
          return action*self.get_price() - 2.5*(action**2)
      return remaining_inventory*(self.get_price() -  self.get_price()) - A*(action**2)

  def PandL(self, action):
      return action*self.get_price() - A*(action**2)

class DQNAgent:
    ''''
    DQN Agent
    Takes as inputs state_size, action_size and TRAINING.
    state_size: Number of different state variables
    action_size: number of different possible actions
    TRAINING: (Bool) Whether the DQN Agent is being Trained (True) or Validated (False)
    '''''
    def __init__(self, state_size, action_size, TRAINING):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=MEMORY)
        self.gamma = 0.99 # discount rate
        self.epsilon = 1.0  if TRAINING else 0.0
        self.epsilon_min = 0.01 if TRAINING else 0.0
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond  = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

        return K.mean(tf.where(cond, squared_loss, quadratic_loss))
    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(20, input_dim=self.state_size, activation='relu'))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=self._huber_loss,
                      optimizer=OPTIMIZER)
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, time, time_constraint_for_execution):
      '''
      Determines the actions of the agent
      '''
      inventory = state[0][1]

      if time == (time_constraint_for_execution - TIME_UNIT):
        action = inventory
        print("last action is " + str(action))
      elif inventory == 0:
        action = 0 #to make it's time consistent with TWAP for comparison purposes
      elif np.random.rand() <= self.epsilon:
          if EGREEDY == 'Binomial':
              n = inventory*INITIAL_INVENTORY
              p = TIME_UNIT/(time_constraint_for_execution - time)
              action = np.random.binomial(n, p)
              action =  np.linspace(0,1,INITIAL_INVENTORY)[action] #scale back action to a normalised action
              print("E-greedy action " + str(action))
          elif EGREEDY == 'Uniform':
              action = random.randrange(self.action_size)
              action = np.linspace(0, 1, INITIAL_INVENTORY)[action]
              print("E-greedy action " + str(action))
      else:
          act_values = self.model.predict(state) # put docstring specifying what act returns
          action = np.argmax(act_values[0])/(INITIAL_INVENTORY-1)
          print("Optimal action " + str(action))

      if action > inventory:
          action = inventory
          print("action>intentory action is " + str(action))
      return round(action,2)

    def act_boundary_conditions(self, state, time, time_constraint_for_execution, index):

        if index == 0:
            if time == (time_constraint_for_execution - TIME_UNIT):
                action = INITIAL_INVENTORY_SCALED
            else:
                action = 0
        elif index == 1:
            if time == 0:
                action = INITIAL_INVENTORY_SCALED
            else:
                action = 0

        return round(action,2)

    def replay(self, batch_size):
      '''
        replay uses the memory values to train the network
      '''
      minibatch = random.sample(self.memory, batch_size)
      for state, action, reward, next_state, done in minibatch:
          target = self.model.predict(state)
          action_range = np.linspace(0, 1, INITIAL_INVENTORY)
          action_index = [i for i in range(len(action_range)) if round(action_range[i].item(), 2) == action][0]
          if done:
              target[0][action_index] = reward
          else:
              # a = self.model.predict(next_state)[0]
              t = self.target_model.predict(next_state)[0]
              target[0][action_index] = reward + self.gamma * np.amax(t)
              # target[0][action] = reward + self.gamma * t[np.argmax(a)]
          self.model.fit(state, target, epochs=1, verbose=0)
      if self.epsilon > self.epsilon_min:
          self.epsilon *= self.epsilon_decay

    def load(self, name):
        '''
        Loads the weights of a trained agent
        '''
        self.model.load_weights(name)

    def save(self, name, PandL_vs_TWAP, PandL_agent, PandL_TWAP):
        '''
        Saves the weights of the trained agent and our results to csvs
        '''
        self.model.save_weights(name)
        state_time = np.array([item[0][0][0] for item in list(self.memory)])
        state_inventory = np.array([item[0][0][1] for item in list(self.memory)])
        actions = np.array([item[1] for item in list(self.memory)])
        rewards = np.array([item[2] for item in list(self.memory)])
        next_state_time = np.array([item[3][0][0] for item in list(self.memory)])
        next_state_inventory = np.array([item[3][0][1] for item in list(self.memory)])
        done = np.array([item[4] for item in list(self.memory)])
        memory_df = pd.DataFrame({'state_time': state_time, 'state_inventory': state_inventory, 'action': actions, 'reward': rewards,'next_state - Inventory': next_state_time, 'next_state - Time': next_state_inventory, 'done':done})
        PandL_df = pd.DataFrame({'PandL_vs_TWAP': PandL_vs_TWAP, 'PandL_agent': PandL_agent, 'PandL_TWAP': PandL_TWAP})
        print(memory_df)
        memory_df.to_csv('DDQNMemory_{}Train={}.csv'.format(EPISODES,TRAINING))
        PandL_df.to_csv('DDQNP&L_{}Train={}.csv'.format(EPISODES, TRAINING))

if __name__ == "__main__":
    run()



