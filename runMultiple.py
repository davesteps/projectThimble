
from tradeEnv import *

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, Permute, Conv1D, MaxPool1D
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory

from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd

# todo - Dates rather than index
# training.py l131
# for i in range(len(names)):
#     if names[i] == 'permute_1_input':
#         arrays[i] = np.expand_dims(arrays[i].squeeze(), 0)
#         # array = arrays[i].squeeze()
#         # arrays[i] = array.squeeze()
#         if len(arrays[i].shape) != 3:
#             arrays[i] = arrays[i].squeeze()

sym =  ['AAPL', 'MSFT', 'GOOGL']
start = '2005-01-01'
end = '2018-03-04'
symbols = data.DataReader(sym, 'google', start, end).loc['Close']

# symbols.plot()

env = MarketMultiple(symbols,max_steps=130)

env.start_index
env.current_index
# mem_len = 1



nb_actions = env.action_space.n

model = Sequential()
model.add(Permute((2, 1), input_shape= env.observation_space.shape))
# model.add(Permute((1,3,2), input_shape= (mem_len,) + env.observation_space.shape))
model.add(Conv1D(64,20))
model.add(Activation('relu'))
model.add(MaxPool1D())
model.add(Conv1D(64,10))
model.add(Activation('relu'))
model.add(MaxPool1D())
model.add(Conv1D(32,5))
model.add(Activation('relu'))
model.add(MaxPool1D())
model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

memory = SequentialMemory(limit=5000,window_length=1)

nb_steps = 5e6
nb_steps_greedy = nb_steps*0.9

# policy = BoltzmannQPolicy(tau=0.05)
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1.,
                              value_min=.1, value_test=.05,nb_steps=nb_steps_greedy)

agent = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
                 nb_steps_warmup=5000, gamma=.99, target_model_update=10000,
                 train_interval=4, delta_clip=1.)
agent.compile(Adam(lr=.00025), metrics=['mae'])
agent.fit(env, nb_steps=nb_steps, log_interval=10000, verbose=1)
agent.save_weights('wghts',overwrite=True)
# agent.load_weights('model8_6c01ba5f786ebd5938ca8e8f75008a63d2fe11bd/dqn_nav2_30x30x3_2conv_1e7_mvhz')

agent.test(env,nb_episodes=1,visualize=False)
env.min_change
env.max_change
env.total_reward


# Finally, evaluate our algorithm for 20 episodes.

#
sym =  ['AAPL', 'MSFT', 'GOOGL']
start = '2005-01-01'
end = '2018-03-04'
symbols = data.DataReader(sym, 'google', start, end).loc['Close']

test_env = MarketMultiple(symbols,test_mode=True)

test_env.current_index
test_env.observation[0,]
float(symbols.iloc[test_env.current_index, [0]])
float(symbols.iloc[test_env.current_index+1, [0]])
test_env.step(0)

((136.7-135.72)/135.72)*100

agent.test(test_env,nb_episodes=1,visualize=False)
test_env.min_change
test_env.max_change
test_env.total_reward

test_env.current_price
test_env.tomorrow_price
test_env.observation
symbols.tail()






