import random
import numpy as np

class Chromosome:
  def __init__(self, chrom=None):

    if chrom is None:
      chrom = {
        'n_food_c': random.randint(0, 1),
        'n_slime_c': random.randint(0, 1),
        # 'x_food_c': random.randint(0, 1),
        # 'y_food_c': random.randint(0, 1),
        # 'x_slime_c': random.randint(0, 1),
        # 'y_slime_c': random.randint(0, 1),
      }

    self.chrom = chrom

  def gaussian_mutation(self, sigma):
    for k, v in self.chrom.items():
      self.chrom[k] = v + np.random.normal(loc=0.0, scale=sigma, size=None)

  def evaluate(self, n_iters):