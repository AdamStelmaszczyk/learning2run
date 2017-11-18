import argparse
import os

from keras import backend as K
from keras.initializers import RandomUniform
from keras.layers import Dense, Flatten, Input, concatenate, Lambda
from keras.models import Sequential, Model
from keras.optimizers import Adam
from osim.env import RunEnv
from osim.http.client import Client
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

import summarize

parser = argparse.ArgumentParser(description='Train or test neural net motor controller')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=False)
parser.add_argument('--submit', dest='submit', action='store_true', default=False)
parser.add_argument('--steps', dest='steps', action='store', default=500000, type=int)
parser.add_argument('--vis', dest='visualize', action='store_true', default=False)
parser.add_argument('--model', dest='model', action='store', default="default")
args = parser.parse_args()

if not (args.train or args.test or args.submit):
    print('No action given, use --train, --test or --submit')
    exit(0)

env = RunEnv(args.visualize)

nb_actions = env.action_space.shape[0]

nallsteps = args.steps

def preprocess(x):
    return K.concatenate([
        x[:,:,0:1] / 360.0,
        x[:,:,1:3],
        x[:,:,3:4] / 360.0, 
        x[:,:,4:6],
        x[:,:,6:18] / 360.0,
        x[:,:,18:19] - x[:,:,1:2],
        x[:,:,19:22],
        x[:,:,28:29] - x[:,:,1:2],
        x[:,:,29:30],
        x[:, :, 30:31] - x[:, :, 1:2],
        x[:, :, 31:32],
        x[:, :, 32:33] - x[:, :, 1:2],
        x[:, :, 33:34],
        x[:, :, 34:35] - x[:, :, 1:2],
        x[:, :, 35:41],
    ], axis=2)

preprocess_layer = Lambda(preprocess, input_shape=(1,) + env.observation_space.shape, output_shape=(35,))
init = RandomUniform(-0.003, 0.003)

actor = Sequential()
actor.add(preprocess_layer)
actor.add(Dense(64, activation=K.tanh, kernel_initializer=init))
#actor.add(BatchNormalization(axis=1))
#actor.add(Dropout(.25))
actor.add(Dense(64, activation=K.tanh, kernel_initializer=init))
actor.add(Dense(nb_actions, activation=K.sigmoid))

print(actor.summary())

action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
reduced_input = preprocess_layer(observation_input)
flattened_observation = Flatten()(reduced_input)
x = concatenate([action_input, flattened_observation])
x = Dense(64, activation=K.tanh, kernel_initializer=init)(x)
#x = Dropout(.25)(x)
x = Dense(64, activation=K.tanh, kernel_initializer=init)(x)
x = Dense(1)(x)
critic = Model(inputs=[action_input, observation_input], outputs=x)

print(critic.summary())

memory = SequentialMemory(limit=1000000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(theta=.15, mu=0., sigma=.2, size=env.noutput)
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, random_process=random_process, gamma=.99, target_model_update=1e-3,
                  delta_clip=1., batch_size=64)
agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

if args.train:
    if os.path.exists(args.model + '_actor'):
        print('Loaded model ' + args.model)
        agent.load_weights(args.model)
    keras_history = agent.fit(env, nb_steps=nallsteps, verbose=2, nb_max_episode_steps=1000)

    print('Saving model ' + args.model)
    agent.save_weights(args.model, overwrite=True)
    print('Saved model ' + args.model)

    with open(args.model + '_history', 'w') as f:
        f.write(str(keras_history.history))
    summarize.plot_diagrams(keras_history.history, args.model)

if args.test:
    agent.load_weights(args.model)
    agent.test(env, nb_episodes=1, nb_max_episode_steps=env.timestep_limit)

if args.submit:
    agent.load_weights(args.model)
    remote_base = 'http://grader.crowdai.org:1729'
    token = '688545d8ba985c174b4f967b40924a43'
    client = Client(remote_base)
    observation = client.env_create(token)
    # The grader runs 3 simulations of at most 1000 steps each. We stop after the last one
    while True:
        [observation, reward, done, info] = client.env_step(agent.forward(observation).tolist())
        print(observation)
        if done:
            observation = client.env_reset()
            if not observation:
                break
    client.submit()
