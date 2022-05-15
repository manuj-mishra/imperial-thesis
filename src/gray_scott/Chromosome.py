import random

import numpy as np

from gray_scott.CAs import CA

class Chromosome:
  def __init__(self, f, k, df, dk):
    self.f = f
    self.k = k
    self.df = df
    self.dk = dk

  @classmethod
  def near_threshold(cls):
    f = np.random.uniform(low=0.0, high=0.0625)
    k = -f + 0.5 * np.sqrt(f)
    df = np.random.uniform(low=0.0, high=0.0625)
    dk = np.random.uniform(low=0.0, high=0.0625)
    return cls(f, k, df, dk)

  def mutate(self):
    self.f += np.random.normal(scale=self.df)
    self.k += np.random.normal(scale=self.dk)

  def loss(self, n_iters, n_steps, max_step, true):
    losses = []
    for _ in range(n_iters):
      pred = CA.patch(self.f, self.k)
      for _ in range(n_steps):
          step_size = random.randint(1, max_step)
          true_active = true.step_from(pred.A, pred.B, step_size)
          pred_active = pred.step(step_size)
          losses.append(np.linalg.norm(pred.state() - true.state()))
          if not true_active and not pred_active:
            break
    return np.mean(losses)
