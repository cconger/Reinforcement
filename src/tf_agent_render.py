import time
import os

from absl import logging
import tensorflow as tf
import numpy as np

import cv2

from tf_agents.utils import common
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import q_network
from tf_agents.environments import tf_py_environment
from tf_agents.drivers import dynamic_episode_driver

from snake_env import SnakeEnv

root_dir = os.path.join('/tf-logs', 'snake')
train_dir = os.path.join(root_dir, 'train')

def run():
    env = tf_py_environment.TFPyEnvironment(SnakeEnv(step_limit=1000))

    ## Needs to be the same network from training
    q_net = q_network.QNetwork(
        env.observation_spec(),
        env.action_spec(),
        conv_layer_params=(),
        fc_layer_params=(256,100),
    )

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)
    global_counter = tf.compat.v1.train.get_or_create_global_step()

    agent = dqn_agent.DqnAgent(
        env.time_step_spec(),
        env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=global_counter,
        gamma=0.95,
        epsilon_greedy=0.1,
        n_step_update=1,
    )

    agent.initialize()

    policy_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(train_dir, 'policy'),
        policy=agent.policy,
        global_step=global_counter,
    )

    policy_checkpointer.initialize_or_restore()

    capture_run(os.path.join(root_dir, "snake" + str(global_counter.numpy()) + ".mp4"), env, agent.policy)

def capture_run(path, eval_env, eval_policy):
    logging.info("Writing video to %s", path)
    #fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fourcc = cv2.VideoWriter_fourcc(*'VP90')
    path = os.path.splitext(path)[0] + '.webm'
    writer = cv2.VideoWriter(path, fourcc, 6.0, (100,100), isColor=False)

    def capture_frame(traj):
        obs = traj.observation.numpy()

        obs = (obs.reshape((10,10,1)) * (255 / 4)).astype(np.uint8)
        obs = cv2.resize(obs, dsize=(100,100), interpolation=cv2.INTER_NEAREST)
        writer.write(obs)


    driver = dynamic_episode_driver.DynamicEpisodeDriver(
            eval_env,
            eval_policy,
            observers=[capture_frame],
            num_episodes=1,
    ).run()

    writer.release()


if __name__ == "__main__":
    run()
