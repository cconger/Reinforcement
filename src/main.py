from collections import deque

import tensorflow as tf
import numpy as np

n_actions = 4

def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=16, kernel_size=4, strides=4, padding='same', activation='relu'),
        tf.keras.layers.Conv2D(filters=32, kernel_size=2, strides=2, padding='same', activation='relu'),
        tf.keras.layers.Conv2D(filters=48, kernel_size=2, strides=1, padding='same', activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=256, activation='relu'),
        tf.keras.layers.Dense(units=n_actions, activation=None),
    ])

    return model

def choose_action(model, obs):
    obs = np.expand_dims(obs, axis=0)

    logits = model.predict(obs)

    prob_weights = tf.nn.softmax(logits).numpy()

    action = np.random.choice(n_actions, size=1, p=prob_weights.flatten())[0]

class Memory:
    def __init__(self, buffer_size=None):
        self._buffer_size = buffer_size
        self.clear()

    def clear(self):
        self.observations = deque(maxlen=self._buffer_size)
        self.actions = deque(maxlen=self._buffer_size)
        self.rewards = deque(maxlen=self._buffer_size)

    def add_memory(self, obs, action, reward):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)

def normalize(x):
    x -= np.mean(x)
    x /= np.std(x)
    return x.astype(np.float32)

def discount_rewards(rewards, gamma=0.95):
    discounted_rewards = np.zeros_like(rewards)
    R = 0
    for t in reversed(range(0, len(rewards))):
        R = R * gamma + rewards[t]
        discounted_rewards[t] = R

    return normalize(discounted_rewards)

train_loss = tr.keras.metrics.Mean('train_loss', dtype=tf.float32)

def train_step(model, optimizer, observations, actions, discounted_rewards):
    with tf.GradientTape() as tape:
        logits = model(observations)

        loss = compute_loss(logits, actions, discounted_rewards)
    train_loss(loss)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

learning_rate = 1e-3
optimizer = tf.keras.optimizers.Adam(learning_rate)
model = create_model()
iterations = 500

memory = Memory(buffer_size=10000)

for i in range(iterations):

