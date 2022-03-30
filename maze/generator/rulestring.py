import random

from generator.evaluate import solution_length
from generator.generate_maze import generate_maze
from generator.media import clear_temp_folders
from generator.region_merge import get_regions, region_merge


class Rulestring:
  def __init__(self, rstring = None):

    if rstring is None:
      rstring = random.randint(0, 2**16 - 1)

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

  def evaluate(self):
    X = generate_maze(self.b, self.s, media=False)
    cells, regions, M = get_regions(X, media=False)
    M, success = region_merge(regions, cells, M, media=False)
    return 1 / solution_length(M) if success else 0


if __name__ == "__main__":
  r = Rulestring()
  print(r.get_rstring())
  r.mutate(0.5)
  print(r.get_rstring())

