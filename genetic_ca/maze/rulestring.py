import random

import numpy as np

from maze.evaluate import dead_ends_and_path_length
from maze.maze import generate_maze, get_regions, region_merge


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
    return ixs

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
    sol_lens = []
    num_ends = []
    for _ in range(n_iters):
      X = generate_maze(self.b, self.s)
      cells, regions, M = get_regions(X)
      M, success = region_merge(regions, cells, M)
      if not success:
        return 0, 0
      _, a, b = dead_ends_and_path_length(M)
      num_ends.append(a)
      sol_lens.append(b)

    sol_lens = np.array(sol_lens)
    num_ends = np.array(num_ends)
    return np.mean(sol_lens), np.mean(num_ends)
