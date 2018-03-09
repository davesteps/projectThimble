
from tradeEnv import *

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, Permute, Conv1D, MaxPool1D
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory

from pandas_datareader import data

import pathlib
from datetime import datetime
import logzero
import logging
import subprocess

import matplotlib.pyplot as plt
import pandas as pd


#########################

sym =  ['AAPL', 'MSFT', 'GOOGL']
start = '2005-01-01'
end = '2018-03-04'

nb_steps = 1e4
nb_steps_greedy = nb_steps*0.9

lr = .00025

#########################

out_root_dir=pathlib.Path('outputs/'+datetime.now().strftime("%Y%m%d_%H%M%S"))

def _get_git_revision_hash():
    hash_bytes = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
    return str(hash_bytes).strip("b'").strip("\\n'")

out_root_dir.mkdir()
log_file = str(out_root_dir / "model.log")
logger = logzero.setup_logger(name=__name__, logfile=log_file, level=logging.DEBUG)
# self.logger.removeHandler(self.logger.handlers[0])  # Remove stream handler (prevents stdout inc. for tests)
logger.info("Starting using version = ""{}".format(_get_git_revision_hash()))
logger.info("Start Date {0}, End Date {1}".format(start,end))
logger.info("Symbols:{}".format(sym))
logger.info("nb_steps:{}".format(nb_steps))
logger.info("nb_steps_greedy:{}".format(nb_steps_greedy))
logger.info("learning_rate:{}".format(lr))


# todo - Dates rather than index
# training.py l131
# for i in range(len(names)):
#     if names[i] == 'permute_1_input':
#         arrays[i] = np.expand_dims(arrays[i].squeeze(), 0)
#         # array = arrays[i].squeeze()
#         # arrays[i] = array.squeeze()
#         if len(arrays[i].shape) != 3:
#             arrays[i] = arrays[i].squeeze()


symbols = data.DataReader(sym, 'google', start, end).loc['Close']

# symbols.plot()

env = MarketMultiple(symbols,max_steps=130)

logger.info("start_index:{}".format(env.start_index))

nb_actions = env.action_space.n

model = Sequential()
model.add(Permute((2, 1), input_shape= env.observation_space.shape))
# model.add(Permute((1,3,2), input_shape= (mem_len,) + env.observation_space.shape))
model.add(Conv1D(64,20))
model.add(Activation('relu'))
model.add(MaxPool1D())
# model.add(Dropout(0.25))
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

with open(log_file,'a') as fh:
    # Pass the file handle in as a lambda function to make it callable
    model.summary(print_fn=lambda x: fh.write(x + '\n'))

memory = SequentialMemory(limit=5000,window_length=1)

# policy = BoltzmannQPolicy(tau=0.05)
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1.,
                              value_min=.1, value_test=.05,nb_steps=nb_steps_greedy)

agent = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
                 nb_steps_warmup=5000, gamma=.99, target_model_update=10000,
                 train_interval=4, delta_clip=1.)
agent.compile(Adam(lr=lr), metrics=['mae'])
agent.fit(env, nb_steps=nb_steps, log_interval=10000, verbose=1)
agent.save_weights(str(out_root_dir)+'/weights',overwrite=True)


logger.info("===============================================================")

# agent.load_weights('model8_6c01ba5f786ebd5938ca8e8f75008a63d2fe11bd/dqn_nav2_30x30x3_2conv_1e7_mvhz')
#
# agent.test(env,nb_episodes=1,visualize=False)
# env.min_change
# env.max_change
# env.total_reward
#
#
#
#
test_env = MarketMultiple(symbols,test_mode=True)

agent.test(test_env,nb_episodes=1,visualize=False)
logger.info("test max change: {}".format(test_env.max_change))
logger.info("test total reward: {}".format(test_env.total_reward))
#
# test_env.current_price
# test_env.tomorrow_price
# test_env.observation
# symbols.tail()






