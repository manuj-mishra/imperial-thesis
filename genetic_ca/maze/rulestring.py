import random

import numpy as np

from maze.maze_ca import MazeCA


class Rulestring:
  def __init__(self, rstring=None):

    if rstring is None:
      rstring = random.randint(0, 2 ** 16 - 1)

    self.b = self._ones(rstring >> 8)
    self.s = self._ones(rstring)
    self.rstring = rstring

  def _ones(self, rstring):
    ixs = []
    for i in range(8, 0, -1):
      if rstring & 1:
        ixs.append(i)
      rstring >>= 1
    return sorted(ixs)

  def get_rstring(self):
    return format(self.rstring, 'b').zfill(16)

  def mutate(self, p):
    mask = 0
    for _ in range(16):
      if random.random() < p:
        mask |= 1
      mask <<= 1

    mask >>= 1
    self.rstring ^= mask

  def evaluate(self, n_iters):
    path_lengths = []
    num_dead_ends = []
    for _ in range(n_iters):
      ca = MazeCA(self.b, self.s)
      success = ca.run()
      if not success:
        return 0, 0
      dead_ends, path_length = ca.dead_ends_and_path_length()
      num_dead_ends.append(dead_ends)
      path_lengths.append(path_length)

    return sum(num_dead_ends) / n_iters, sum(path_lengths) / n_iters
