import random

import numpy as np
from new_slime.slime_ca import SlimeCA


class Chromosome:
  def __init__(self, b=None, s=None):
    if b is None:
      b = [random.randint(1, 2 ** 9 - 1) for _ in range(9)]
    if s is None:
      s = [random.randint(1, 2 ** 9 - 1) for _ in range(9)]
    self.b = b
    self.s = s

  def mutate_chrom(self, p):
    for i in range(9):
      self.b[i] = self.b[i] ^ self.gene_mask(p)
      self.s[i] = self.s[i] ^ self.gene_mask(p)

  def gene_mask(self, p):
    mask = 0
    for _ in range(9):
      if random.random() < p:
        mask |= 1
      mask <<= 1

    mask >>= 1
    return mask

  def evaluate(self, n_iters):
    foods = []
    sizes = []
    for _ in range(n_iters):
      ca = SlimeCA(self.b, self.s)
      ca.run()
      food, size = ca.metrics()
      foods.append(food)
      sizes.append(size)

    return np.mean(foods), np.mean(sizes)

  def id(self):
    return hash((tuple(self.b), tuple(self.s)))
