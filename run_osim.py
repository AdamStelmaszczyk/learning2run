import argparse
import logging
import os
import subprocess
from datetime import datetime
from random import randint

import gym
import tensorflow as tf
from baselines import logger
from baselines.common import tf_util as U
from baselines.common.mpi_fork import mpi_fork
from baselines.pposgd import mlp_policy, pposgd_simple
from baselines.pposgd.pposgd_simple import preprocess
from osim.env import RunEnv
from osim.redis.client import Client
from mpi4py import MPI
import opensim

import summarize

parser = argparse.ArgumentParser(description='Train or test neural net motor controller')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--repeat', dest='repeat', action='store_true', default=False)
parser.add_argument('--original', dest='original', action='store_true', default=False)
parser.add_argument('--obstacles', dest='obstacles', action='store', default=10, type=int)
parser.add_argument('--test', dest='test', action='store_true', default=False)
parser.add_argument('--submit', dest='submit', action='store_true', default=False)
parser.add_argument('--steps', dest='steps', action='store', default=500000, type=int)
parser.add_argument('--batch', dest='batch', action='store', default=2048, type=int)
parser.add_argument('--max_steps', dest='max_steps', action='store', default=1000, type=int)
parser.add_argument('--optim_batch', dest='optim_batch', action='store', default=64, type=int)
parser.add_argument('--size', dest='size', action='store', default=64, type=int)
parser.add_argument('--layers', dest='layers', action='store', default=2, type=int)
parser.add_argument('--cores', dest='cores', action='store', default=4, type=int)
parser.add_argument('--seed', dest='seed', action='store', default=randint(1, 1000000), type=int)
parser.add_argument('--ent', dest='ent', action='store', default=0.0, type=float)
parser.add_argument('--stepsize', dest='stepsize', action='store', default=0.0003, type=float)
parser.add_argument('--clip', dest='clip', action='store', default=0.2, type=float)
parser.add_argument('--gamma', dest='gamma', action='store', default=0.99, type=float)
parser.add_argument('--keep', dest='keep', action='store', default=1.0, type=float)
parser.add_argument('--epochs', dest='epochs', action='store', default=10, type=int)
parser.add_argument('--vis', dest='visualize', action='store_true', default=False)
parser.add_argument('--verbose', dest='verbose', action='store_true', default=False)
parser.add_argument('--model', dest='model', action='store', default='default')
parser.add_argument('--activation', dest='activation', action='store', default='tanh')
parser.add_argument('--schedule', dest='schedule', action='store', default='linear')
args = parser.parse_args()

if not (args.train or args.test or args.submit):
    print('No action given, use --train, --test or --submit')
    exit(0)

gym.logger.setLevel(logging.WARN)


def time():
    return datetime.now().strftime('%H:%M:%S')


def policy_fn(name, ob_space, ac_space):
    return mlp_policy.MlpPolicy(
        name=name,
        ob_space=ob_space,
        ac_space=ac_space,
        hid_size=args.size,
        num_hid_layers=args.layers,
        activation=args.activation,
        keep=args.keep,
    )


def plot_history(h, iteration=0):
    if iteration % 25 == 0 and MPI.COMM_WORLD.Get_rank() == 0:
        data = {
            "EpRewMean": h["EpRewMean"],
            "EpLenMean": h["EpLenMean"],
            "loss_pol_surr": h["loss_pol_surr"],
            "loss_kl": h["loss_kl"],
        }
        summarize.plot_diagrams(data, args.model)


def load_model(iteration=0):
    if iteration <= 1 and os.path.exists(args.model + '.meta'):
        tf.train.Saver().restore(session, args.model)
        if MPI.COMM_WORLD.Get_rank() == 0:
            print('Loaded model %s' % args.model)
        return True
    if iteration <= 1 and MPI.COMM_WORLD.Get_rank() == 0:
        print('Model %s not found' % args.model)
    return False


def save_model(iteration=0):
    if iteration % 25 == 0 and MPI.COMM_WORLD.Get_rank() == 0:
        print('Saving model ' + args.model)
        saver = tf.train.Saver()
        saver.save(session, args.model)
        print('Saved model ' + args.model + ' at ' + time())


def on_iteration_start(local_vars, global_vars):
    on_iteration_start.iteration += 1
    load_model(on_iteration_start.iteration)
    plot_history(local_vars['history'], on_iteration_start.iteration)
    save_model(on_iteration_start.iteration)


on_iteration_start.iteration = 0

whoami = mpi_fork(args.cores)
if whoami == 'parent':
    exit(0)

session = U.single_threaded_session()
session.__enter__()
logger.session().__enter__()

env = RunEnv(args.visualize, max_obstacles=args.obstacles, original_reward=args.original)
env.spec.timestep_limit = args.max_steps
if args.visualize:
    vis = env.osim_model.model.updVisualizer().updSimbodyVisualizer()
    vis.setBackgroundType(vis.GroundAndSky)
    vis.setShowFrameNumber(True)
    vis.zoomCameraToShowAllGeometry()
    vis.setCameraFieldOfView(1)

if args.train:
    history = pposgd_simple.learn(
        env,
        policy_fn,
        max_timesteps=args.steps,
        timesteps_per_batch=args.batch,
        clip_param=args.clip,
        entcoeff=args.ent,
        optim_epochs=args.epochs,
        optim_stepsize=args.stepsize,
        optim_batchsize=args.optim_batch,
        adam_epsilon=1e-5,
        gamma=args.gamma,
        lam=0.95,
        schedule=args.schedule,
        callback=on_iteration_start,
        verbose=args.verbose,
    )

    env.close()

    if MPI.COMM_WORLD.Get_rank() == 0:
        plot_history(history)
        save_model()

    if args.repeat:
        cmd = 'python run_osim.py --repeat --train --model %s --steps %s --size %s' % (args.model, args.steps, args.size)
        subprocess.call(cmd.split(' '))

if args.test:
    observation = env.reset()
    observation = preprocess(observation, step=1, verbose=args.verbose)
    pi = policy_fn('pi', env.observation_space, env.action_space)

    if not load_model():
        exit(0)

    done = False
    total = 0
    steps = 0
    while not done:
        action = pi.act(True, observation)[0]
        observation, reward, done, info = env.step(action)
        if args.visualize:
            vis.pointCameraAt(opensim.Vec3(observation[1], 0, 0), opensim.Vec3(0, 1, 0))
        observation = preprocess(observation, step=steps + 2, verbose=args.verbose)
        total += reward
        env.render()
        steps += 1
    print('Total reward: %s' % total)
    print('Steps: %s' % steps)

if args.submit:
    token = '688545d8ba985c174b4f967b40924a43'
    client = Client()
    observation = client.env_create()
    observation = preprocess(observation, step=1)
    pi = policy_fn('pi', env.observation_space, env.action_space)

    if not load_model():
        exit(0)

    # The grader runs 3 simulations of at most 1000 steps each. We stop after the last one
    final_steps = []
    final_returns = []
    total = 0
    steps = 0
    while True:
        action = pi.act(True, observation)[0].tolist()
        observation, reward, done, info = client.env_step(action)
        observation = preprocess(observation, step=steps + 2, verbose=True)
        total += reward
        steps += 1
        if done:
            final_steps.append(steps)
            final_returns.append(total)
            observation = client.env_reset()
            if not observation:
                break
            total = 0
            steps = 0
            observation = preprocess(observation, step=1, verbose=True)
    client.submit()
    print("Steps: %s" % final_steps)
    print("Returns: %s" % final_returns)
