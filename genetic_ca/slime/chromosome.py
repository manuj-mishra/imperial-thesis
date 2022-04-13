import random
import numpy as np

from slime.slime_ca import SlimeCA


class Chromosome:
  def __init__(self, coefs=None):
    if coefs is None:
      self.coefs = {
        'b_food_n': random.randint(0, 1),
        'b_food_x': random.randint(0, 1),
        'b_food_y': random.randint(0, 1),
        'b_slime_n': random.randint(0, 1),
        'b_slime_x': random.randint(0, 1),
        'b_slime_y': random.randint(0, 1),
        's_food_n': random.randint(0, 1),
        's_food_x': random.randint(0, 1),
        's_food_y': random.randint(0, 1),
        's_slime_n': random.randint(0, 1),
        's_slime_x': random.randint(0, 1),
        's_slime_y': random.randint(0, 1),
        'threshold': random.randint(0, 1)
      }

  def id(self):
    return hash(tuple(self.coefs.values()))

  # NOT USED
  def total_gaussian_mutation(self, sigma):
    for k, v in self.coefs.items():
      v_new = v + np.random.normal(loc=0.0, scale=sigma, size=None)
      v_new = max(0, v_new)
      v_new = min(1, v_new)
      self.coefs[k] = v_new

  def single_non_uniform_mutation(self, curr_epoch, max_epoch):
    k, v = random.choice(list(self.coefs.items()))
    r1 = random.random()
    r2 = random.random()
    f = r2 * (1 - curr_epoch / max_epoch)
    if r1 > 0.5:
      self.coefs[k] = (1 - v) * f
    else:
      self.coefs[k] = v * f

  def evaluate(self, n_iters):
    food_eaten = []
    slime_size = []
    for _ in range(n_iters):
      ca = SlimeCA(self)
      ca.run()
      food_eaten.append(ca.food_reached())
      slime_size.append(ca.slime_size())
    return sum(food_eaten) / n_iters, sum(slime_size) / n_iters