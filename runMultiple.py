
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
for i in range(len(names)):
    if names[i] == 'permute_1_input':
        arrays[i] = np.expand_dims(arrays[i].squeeze(), 0)
        # array = arrays[i].squeeze()
        # arrays[i] = array.squeeze()
        if len(arrays[i].shape) != 3:
            arrays[i] = arrays[i].squeeze()

sym =  ['AAPL', 'MSFT', 'GOOGL']

# User pandas_reader.data.DataReader to load the desired data. As simple as that.
start = '2005-01-01'
end = '2018-03-01'
symbols = data.DataReader(sym, 'google', start, end).loc['Close']

# symbols.plot()

env = MarketMultiple(symbols,max_steps=130)
env.observation.shape

# mem_len = 1

nb_actions = env.action_space.n

input_shape = env.observation_space.shape
input_shape
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

# policy = BoltzmannQPolicy(tau=0.05)
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1.,
                              value_min=.1, value_test=.05,nb_steps=9e5)

agent = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
                 nb_steps_warmup=5000, gamma=.99, target_model_update=10000,
                 train_interval=4, delta_clip=1.)
agent.compile(Adam(lr=.00025), metrics=['mae'])
agent.fit(env, nb_steps=1e6, log_interval=10000,verbose=1)
# agent.save_weights('wghts',overwrite=True)
# agent.load_weights('model8_6c01ba5f786ebd5938ca8e8f75008a63d2fe11bd/dqn_nav2_30x30x3_2conv_1e7_mvhz')

agent.test(env,nb_episodes=1,visualize=False)
env.episode_change
env.total_reward
env.closed_positions


# Finally, evaluate our algorithm for 20 episodes.
test_env = Market(symbols[sym],test_mode=True)
test_env.current_index
test_env.current_price
agent.test(test_env,nb_episodes=1,visualize=False)
test_env.current_index
test_env.current_price
test_env.episode_change
test_env.total_reward
test_env.closed_positions
test_env.position
test_env.position_open_price



test_env.position
test_env.step(2)
