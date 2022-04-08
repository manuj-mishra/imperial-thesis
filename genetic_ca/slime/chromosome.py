import random
import numpy as np

from slime.slime_ca import SlimeCA


class Chromosome:
  def __init__(self, b_coefs=None, s_coefs=None):

    if b_coefs is None:
      self.b_coefs = {
        'n_food_c': random.randint(0, 1),
        'n_slime_c': random.randint(0, 1),
        # 'x_food_c': random.randint(0, 1),
        # 'y_food_c': random.randint(0, 1),
        # 'x_slime_c': random.randint(0, 1),
        # 'y_slime_c': random.randint(0, 1),
      }

    if s_coefs is None:
      self.s_coefs = {
        'n_food_c': random.randint(0, 1),
        'n_slime_c': random.randint(0, 1),
        # 'x_food_c': random.randint(0, 1),
        # 'y_food_c': random.randint(0, 1),
        # 'x_slime_c': random.randint(0, 1),
        # 'y_slime_c': random.randint(0, 1),
      }

  def gaussian_mutation(self, sigma):
    for c_dict in (self.b_coefs, self.s_coefs):
      for k, v in c_dict.items():
        v_new = v + np.random.normal(loc=0.0, scale=sigma, size=None)
        v_new = max(0, v_new)
        v_new = min(1, v_new)
        c_dict[k] = v_new

  def evaluate(self, n_iters):
    food_reached = []
    slime_size = []
    for _ in range(n_iters):
      ca = SlimeCA(self)
      ca.run()
      food_reached.append(ca.food_reached())
      slime_size.append(ca.slime_size())
    return sum(food_reached) / n_iters, sum(slime_size) / n_iters

  def __str__(self):
    return str(self.b_coefs.values()) + "_" + str(self.s_coefs.values())
