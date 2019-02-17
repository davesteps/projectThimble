
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
# t

sym =  'BRSC'#, 'MSFT', 'GOOGL']

# User pandas_reader.data.DataReader to load the desired data. As simple as that.
start = '2000-01-01'
end = '2018-03-01'
panel_data = data.DataReader([sym], 'google', start, end)

symbols = panel_data.ix['Close'][sym]

symbols.plot()

env = Market(symbols)

env.max_steps
env.current_index
env.current_price

mem_len = 1

nb_actions = env.action_space.n

# Next, we build a very simple model.
# Convolution1D(nb_filter=nb_filter, filter_length=filter_length, activation='relu',
#               input_shape=(window_size, nb_input_series)),
# MaxPooling1D(),  # Downsample the output of convolution by 2X.
# Convolution1D(nb_filter=nb_filter, filter_length=filter_length, activation='relu'),
# MaxPooling1D(),
# Flatten(),
# Dense(nb_outputs, activation='linear'),

model = Sequential()
model.add(Permute((2, 1), input_shape= env.observation_space.shape))
model.add(Conv1D(32,10))
model.add(Activation('relu'))
model.add(MaxPool1D())
model.add(Conv1D(32,5))
model.add(Activation('relu'))
model.add(MaxPool1D())
# model.add(Permute((2, 3, 1), input_shape=(mem_len,) + env.observation_space.shape))
# model.add(Conv2D(32, (8, 8), strides=(4, 4)))
# model.add(Activation('relu'))
# model.add(Conv2D(64, (4, 4), strides=(2, 2)))
# model.add(Activation('relu'))
# model.add(Conv2D(64, (3, 3), strides=(1, 1)))
# model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=500000, window_length=1)

# policy = BoltzmannQPolicy(tau=0.05)
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1.,
                              value_min=.1, value_test=.05,nb_steps=1e5)

agent = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
                 nb_steps_warmup=50000, gamma=.99, target_model_update=10000,
                 train_interval=4, delta_clip=1.)
agent.compile(Adam(lr=.00025), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
agent.fit(env, nb_steps=1e5, log_interval=10000,verbose=2)
agent.save_weights('wghts',overwrite=True)
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
