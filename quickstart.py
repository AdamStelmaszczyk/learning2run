# Copyright 2017 reinforce.io. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import argparse

from osim.env import *
from tensorforce import Configuration
from tensorforce.agents import TRPOAgent, PPOAgent
from tensorforce.core.networks import layered_network_builder
from tensorforce.execution import Runner

from tensor_force_env import TensorForceEnv

parser = argparse.ArgumentParser(description='Train or test neural net motor controller')
parser.add_argument('--episodes', dest='episodes', action='store', default=10000, type=int)
parser.add_argument('--vis', dest='visualize', action='store_true', default=False)
parser.add_argument('--model', dest='model', action='store', default='tensorforce')
args = parser.parse_args()

env = RunEnv(args.visualize)
env = TensorForceEnv(env)

trpo_agent = TRPOAgent(config=Configuration(
    log_level='debug',
    batch_size=64,
    baseline=dict(
        type='mlp',
        size=64,
        hidden_layers=2,
        epochs=5,
        update_batch_size=64,
    ),
    generalized_advantage_estimation=True,
    normalize_advantage=False,
    gae_lambda=0.97,
    max_kl_divergence=0.005,
    cg_iterations=20,
    cg_damping=0.01,
    ls_max_backtracks=20,
    ls_override=False,
    states=env.states,
    actions=env.actions,
    network=layered_network_builder([
        dict(type='dense', size=64),
        dict(type='dense', size=64),
    ])
))

ppo_agent = PPOAgent(config=Configuration(
    log_level='debug',
    batch_size=64,
    entropy_penalty=0.01,
    loss_clipping=0.1,
    epochs=10,
    optimizer_batch_size=64,
    learning_rate=0.0003,
    states=env.states,
    actions=env.actions,
    network=layered_network_builder([
        dict(type='dense', size=64),
        dict(type='dense', size=64),
    ])
))

agent = trpo_agent

if os.path.exists(args.model + '.meta'):
    agent.load_model(args.model)
    print "Loaded model " + args.model

runner = Runner(agent=agent, environment=env)


def episode_finished(r):
    print("Finished episode {ep} after {ts} timesteps (reward: {reward})".format(
        ep=r.episode,
        ts=r.timestep,
        reward=r.episode_rewards[-1]
    ))
    return True


runner.run(episodes=args.episodes, max_timesteps=1000, episode_finished=episode_finished)

print("Learning finished. Total episodes: {ep}. Average reward of last 100 episodes: {ar}.".format(
    ep=runner.episode,
    ar=np.mean(runner.episode_rewards[-100:])
))

agent.save_model(args.model)
print("Saved model " + args.model)
