import numpy as np

from snake_game import SnakeGame

from tf_agents.environments import py_environment
from tf_agents.trajectories import time_step as ts
from tf_agents.specs import array_spec

class SnakeEnv(py_environment.PyEnvironment):
    def __init__(self, step_limit=None):
        self._action_spec = array_spec.BoundedArraySpec(
                shape=(), dtype=np.int32, minimum=0, maximum=3, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
                shape=(100,), dtype=np.int32, minimum=0, name="observation")
        self._game = SnakeGame(size=10, no_grow=True)
        self._episode_ended = False
        self._step_limit = step_limit
        self._step_count = 0

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec
    
    def _reset(self):
        self._episode_ended = False
        self._step_count = 0
        obs = self._game.reset()
        return ts.restart(obs.flatten())
    
    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        self._step_count += 1
        obs, reward, terminal = self._game.step(action)
        obs = obs.flatten()

        if terminal:
            self._episode_ended = True

        if self._step_limit is not None and self._step_count > self._step_limit:
            self._episode_ended = True

        if self._episode_ended:
            return ts.termination(obs, reward)

        return ts.transition(obs, reward, discount=1.0)

