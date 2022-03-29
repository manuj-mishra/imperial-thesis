import random

from generator.evaluate import solution_length
from generator.generate_maze import generate_maze
from generator.media import clear_temp_folders
from generator.region_merge import get_regions, region_merge


class Rulestring:
  def __init__(self):

    birth = []
    survival = []

    for i in range(1,9):
      if random.random() < 0.5:
        birth.append(i)

    for i in range(1,9):
      if random.random() < 0.5:
        survival.append(i)

    self.b = birth
    self.s = survival
    self.rstring = self._rstring(birth, survival)

  def _rstring(self, b, s):
    rstring = 0

    for i in range(1,9):
      if i in b:
        rstring |= 1
      rstring <<= 1

    for i in range(1,9):
      if i in s:
        rstring |= 1
      rstring <<= 1

    rstring >>= 1
    return rstring

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

  def evaluate(self):
    X = generate_maze(self.b, self.s, media=False)
    cells, regions, M = get_regions(X, media=False)
    M = region_merge(regions, cells, M, media=False)
    return solution_length(M)


if __name__ == "__main__":
  r = Rulestring()
  print(r.get_rstring())
  r.mutate(0.5)
  print(r.get_rstring())

