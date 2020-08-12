import numpy as np
import random

# Observation Values
EMPTY = 0
WALL = 1
TREAT = 2
SNAKE_HEAD = 3
SNAKE = 4

# Action Space
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

ACTION_TO_NAME = {
    UP: "UP",
    RIGHT: "RIGHT",
    DOWN: "DOWN",
    LEFT: "LEFT",
}

RENDERMAP = {
    EMPTY: ' ',
    WALL: '█',
    TREAT: 'X',
    SNAKE: 'O',
    SNAKE_HEAD: 'H',
}

class SnakeGame(object):
  def __init__(self, size=10, no_grow=False):
    if size < 3:
      raise ValueError("size must be larger than 3")
    self._size = size
    self._no_grow = no_grow
    self.reset()
  
  def reset(self):
    self._grid = np.zeros(shape=(self._size, self._size), dtype=np.int32)
    self._prev_action = None

    # Paint all the walls
    for i in range(self._size):
      self._grid[0][i] = WALL
      self._grid[self._size-1][i] = WALL
      self._grid[i][0] = WALL
      self._grid[i][self._size-1] = WALL

    self._score = 0
    self._length = 0
    startx = random.randrange(self._size - 2) + 1
    starty = random.randrange(self._size - 2) + 1
    self._headpos = (startx, starty)
    self._tailpos = (startx, starty)
    self._grid[starty][startx] = SNAKE_HEAD
    self._place_treat()
    return self._observe()
  
  def _place_treat(self):
    while True:
      rx = random.randrange(self._size - 2) + 1
      ry = random.randrange(self._size - 2) + 1
      if self._grid[ry][rx] == 0:
        self._grid[ry][rx] = TREAT
        break

  def _observe(self):
    return self._hide_detail()
    #return self._hide_detail().astype(np.float32).reshape(self._size, self._size, 1)

  def _hide_detail(self):
    return np.where(self._grid <= SNAKE_HEAD, self._grid, SNAKE)
  
  def render_frame(self):
    grid = self._hide_detail()
    edge = ''.join(['═' for i in range(self._size)])
    return '\n'.join(["╔" + edge + "╗", '\n'.join(['║' + ''.join([RENDERMAP[v] for v in row]) + '║' for row in grid]), "╚" + edge + "╝"])
    
  def render(self):
    grid = self._hide_detail()
    return '\n'.join([''.join([RENDERMAP[v] for v in row]) for row in grid])
  
  def step(self, direction):
    if direction == UP:
      target_pos = (self._headpos[0], self._headpos[1] - 1)
    elif direction == RIGHT:
      target_pos = (self._headpos[0]+1, self._headpos[1])
    elif direction == DOWN:
      target_pos = (self._headpos[0], self._headpos[1]+1)
    elif direction == LEFT:
      target_pos = (self._headpos[0]-1, self._headpos[1])
    else:
      raise Exception("Invalid movement direction")

    self._prev_action = direction

    if min(target_pos) < 0 or max(target_pos) >= self._size:
      return (self._observe(), -1., True)
    
    if self._grid[target_pos[1]][target_pos[0]] == WALL:
      return (self._observe(), -1., True)

    if self._grid[target_pos[1]][target_pos[0]] >= SNAKE_HEAD:
      return (self._observe(), -1., True)

    self._headpos = target_pos

    ate_treat = False
    if self._grid[target_pos[1]][target_pos[0]] == TREAT:
      ate_treat = True
      if not self._no_grow:
        self._length += 1
    
    next = np.where(self._grid < SNAKE_HEAD, self._grid, self._grid + 1)
    self._grid = np.where(next <= SNAKE_HEAD + self._length, next, EMPTY)
    self._grid[target_pos[1]][target_pos[0]] = SNAKE_HEAD

    if ate_treat:
      self._score += 1
      self._place_treat()

    return (self._observe(), 1. if ate_treat else 0., False)

if __name__ == "__main__":
  game = SnakeGame(size=10, no_grow=True)
  done = False
  while not done:
    print(game.render())
    move = input("Move: ")
    if move[0] == 'u':
      obs, reward, done = game.step(0)
    elif move[0] == 'r':
      obs, reward, done = game.step(1)
    elif move[0] == 'd':
      obs, reward, done = game.step(2)
    elif move[0] == 'l':
      obs, reward, done = game.step(3)
    
    print("reward:", reward)

  print("GAME OVER")
