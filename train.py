from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import tensorflow as tf

from agents import tools
from agents.scripts import configs
from agents.scripts import utility
from gym.envs import register
from osim.env import RunEnv


def _create_environment(config):
  """Constructor for an instance of the environment.

  Args:
    config: Object providing configurations via attributes.

  Returns:
    Wrapped OpenAI Gym environment.
  """
  env = RunEnv(visualize=False)
  if config.max_length:
    env = tools.wrappers.LimitDuration(env, config.max_length)
  env = tools.wrappers.RangeNormalize(env)
  env = tools.wrappers.ClipAction(env)
  env = tools.wrappers.ConvertTo32Bit(env)
  return env


def _define_loop(graph, logdir, train_steps, eval_steps):
  """Create and configure a training loop with training and evaluation phases.

  Args:
    graph: Object providing graph elements via attributes.
    logdir: Log directory for storing checkpoints and summaries.
    train_steps: Number of training steps per epoch.
    eval_steps: Number of evaluation steps per epoch.

  Returns:
    Loop object.
  """
  loop = tools.Loop(
      logdir, graph.step, graph.should_log, graph.do_report,
      graph.force_reset)
  loop.add_phase(
      'train', graph.done, graph.score, graph.summary, train_steps,
      report_every=None,
      log_every=train_steps // 2,
      checkpoint_every=None,
      feed={graph.is_training: True})
  loop.add_phase(
      'eval', graph.done, graph.score, graph.summary, eval_steps,
      report_every=eval_steps,
      log_every=eval_steps // 2,
      checkpoint_every=10 * eval_steps,
      feed={graph.is_training: False})
  return loop


def train(config, env_processes):
  """Training and evaluation entry point yielding scores.

  Resolves some configuration attributes, creates environments, graph, and
  training loop. By default, assigns all operations to the CPU.

  Args:
    config: Object providing configurations via attributes.
    env_processes: Whether to step environments in separate processes.

  Yields:
    Evaluation scores.
  """
  tf.reset_default_graph()
  with config.unlocked:
    config.policy_optimizer = getattr(tf.train, config.policy_optimizer)
    config.value_optimizer = getattr(tf.train, config.value_optimizer)
  if config.update_every % config.num_agents:
    tf.logging.warn('Number of agents should divide episodes per update.')
  with tf.device('/cpu:0'):
    batch_env = utility.define_batch_env(
        lambda: _create_environment(config),
        config.num_agents, env_processes)
    graph = utility.define_simulation_graph(
        batch_env, config.algorithm, config)
    loop = _define_loop(
        graph, config.logdir,
        config.update_every * config.max_length,
        config.eval_episodes * config.max_length)
    total_steps = int(
        config.steps / config.update_every *
        (config.update_every + config.eval_episodes))

  # Exclude episode related variables since the Python state of environments is
  # not checkpointed and thus new episodes start after resuming.
  saver = utility.define_saver(exclude=(r'.*_temporary/.*',))
  sess_config = tf.ConfigProto(allow_soft_placement=True)
  sess_config.gpu_options.allow_growth = True
  with tf.Session(config=sess_config) as sess:
    utility.initialize_variables(sess, saver, config.logdir)
    for score in loop.run(sess, saver, total_steps):
      yield score
  batch_env.close()


register(
    id='Osim-v0',
    entry_point='osim.env.run:RunEnv',
    trials=100,
    max_episode_steps=1000,
)

logdir = datetime.datetime.now().strftime('%H-%M-%S')

try:
    config = utility.load_config(logdir)
except IOError:
    config = tools.AttrDict(getattr(configs, 'osim')())
    config = utility.save_config(config, logdir)

for score in train(config, env_processes=False):
    tf.logging.info('Score {}.'.format(score))