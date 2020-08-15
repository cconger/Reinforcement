import time
import os

from absl import logging
import tensorflow as tf
import numpy as np

from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import q_network
from tf_agents.environments import tf_py_environment
from tf_agents.metrics import tf_metrics
from tf_agents.policies import random_tf_policy
from tf_agents.drivers import dynamic_step_driver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
from tf_agents.eval import metric_utils

from snake_env import SnakeEnv
from tf_agent_render import capture_run

num_iterations = 2000000
batch_size = 64
learning_rate = 1e-3
log_interval = 20

num_eval_episodes = 10
eval_interval = 1000
train_steps_per_iteration = 1

initial_collect_episodes = 100
initial_collect_steps = 10000
collect_episodes_per_iteration = 1
replay_buffer_max_length = 100000
collect_steps_per_iteration = 1

train_checkpoint_interval = 5000
policy_checkpoint_interval = 5000
rb_checkpoint_interval = 5000
capture_interval = 10000

def run():
  tf_env = tf_py_environment.TFPyEnvironment(SnakeEnv())
  eval_env = tf_py_environment.TFPyEnvironment(SnakeEnv(step_limit=50))

  q_net = q_network.QNetwork(
      tf_env.observation_spec(),
      tf_env.action_spec(),
      conv_layer_params=(),
      fc_layer_params=(512,256,128),
  )

  optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
  global_counter = tf.compat.v1.train.get_or_create_global_step()

  agent = dqn_agent.DqnAgent(
      tf_env.time_step_spec(),
      tf_env.action_spec(),
      q_network=q_net,
      optimizer=optimizer,
      td_errors_loss_fn=common.element_wise_squared_loss,
      train_step_counter=global_counter,
      gamma=0.95,
      epsilon_greedy=0.1,
      n_step_update=1,
  )

  root_dir = os.path.join('/tf-logs', 'snake')
  train_dir = os.path.join(root_dir, 'train')
  eval_dir = os.path.join(root_dir, 'eval')

  agent.initialize()

  train_metrics = [
      tf_metrics.NumberOfEpisodes(),
      tf_metrics.EnvironmentSteps(),
      tf_metrics.AverageReturnMetric(),
      tf_metrics.AverageEpisodeLengthMetric(),
  ]


  replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
      data_spec=agent.collect_data_spec,
      batch_size=tf_env.batch_size,
      max_length=replay_buffer_max_length,
  )

  collect_driver = dynamic_step_driver.DynamicStepDriver(
    tf_env,
    agent.collect_policy,
    observers=[replay_buffer.add_batch] + train_metrics,
    num_steps=collect_steps_per_iteration,
  )

  train_checkpointer = common.Checkpointer(
      ckpt_dir=train_dir,
      agent=agent,
      global_step=global_counter,
      metrics=metric_utils.MetricsGroup(train_metrics, 'train_metrics'),
  )

  policy_checkpointer = common.Checkpointer(
      ckpt_dir=os.path.join(train_dir, 'policy'),
      policy=agent.policy,
      global_step=global_counter,
  )

  rb_checkpointer = common.Checkpointer(
      ckpt_dir=os.path.join(train_dir, 'replay_buffer'),
      max_to_keep=1,
      replay_buffer=replay_buffer,
  )

  train_checkpointer.initialize_or_restore()
  rb_checkpointer.initialize_or_restore()

  collect_driver.run = common.function(collect_driver.run)
  agent.train = common.function(agent.train)

  random_policy = random_tf_policy.RandomTFPolicy(
      tf_env.time_step_spec(), tf_env.action_spec())

  if replay_buffer.num_frames() >= initial_collect_steps:
      logging.info("We loaded memories, not doing random seed")
  else:
    logging.info("Capturing %d steps to seed with random memories", initial_collect_steps)

    dynamic_step_driver.DynamicStepDriver(
        tf_env,
        random_policy,
        observers=[replay_buffer.add_batch] + train_metrics,
        num_steps=initial_collect_steps
    ).run()

  train_summary_writer = tf.summary.create_file_writer(train_dir)
  train_summary_writer.set_as_default()

  avg_returns = []
  avg_return_metric = tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes)
  eval_metrics = [
      avg_return_metric,
      tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes),
  ]
  logging.info("Running initial evaluation")
  results = metric_utils.eager_compute(
      eval_metrics,
      eval_env,
      agent.policy,
      num_episodes=num_eval_episodes,
      train_step=global_counter,
      summary_writer=tf.summary.create_file_writer(eval_dir),
      summary_prefix='Metrics',
  )
  avg_returns.append((global_counter.numpy(), avg_return_metric.result().numpy()))
  metric_utils.log_metrics(eval_metrics)

  time_step = None
  policy_state = agent.collect_policy.get_initial_state(tf_env.batch_size)

  timed_at_step = global_counter.numpy()
  time_acc = 0

  dataset = replay_buffer.as_dataset(
      num_parallel_calls=3,
      sample_batch_size=batch_size,
      num_steps=2
  ).prefetch(3)

  iterator = iter(dataset)

  @common.function
  def train_step():
    experience, _ = next(iterator)
    return agent.train(experience)

  for _ in range(num_iterations):
    start_time = time.time()
    time_step, policy_state = collect_driver.run(
        time_step=time_step,
        policy_state=policy_state,
    )

    for _ in range(train_steps_per_iteration):
      train_loss = train_step()
    time_acc += time.time() - start_time

    step = global_counter.numpy()

    if step % log_interval == 0:
      logging.info("step = %d, loss = %f", step, train_loss.loss)
      steps_per_sec = (step - timed_at_step) / time_acc
      logging.info("%.3f steps/sec", steps_per_sec)
      timed_at_step = step
      time_acc = 0

    for train_metric in train_metrics:
      train_metric.tf_summaries(
          train_step=global_counter, step_metrics=train_metrics[:2])

    if step % train_checkpoint_interval == 0:
      train_checkpointer.save(global_step=step)

    if step % policy_checkpoint_interval == 0:
      policy_checkpointer.save(global_step=step)

    if step % rb_checkpoint_interval == 0:
      rb_checkpointer.save(global_step=step)

    if step % capture_interval == 0:
      print("Capturing run:")
      capture_run(os.path.join(root_dir, "snake" + str(step) + ".mp4"), eval_env, agent.policy)
    
    if step % eval_interval == 0:
      print("EVALUTION TIME:")
      results = metric_utils.eager_compute(
          eval_metrics,
          eval_env,
          agent.policy,
          num_episodes=num_eval_episodes,
          train_step=global_counter,
          summary_writer=tf.summary.create_file_writer(eval_dir),
          summary_prefix='Metrics',
      )
      metric_utils.log_metrics(eval_metrics)
      avg_returns.append((global_counter.numpy(), avg_return_metric.result().numpy()))


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    run()
