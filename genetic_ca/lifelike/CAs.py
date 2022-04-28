import random
import numpy as np
from scipy.signal import convolve2d

GRID_SIZE = 32


class CA:
  def __init__(self, X, B, S):
    self.X = X
    self.B = B
    self.S = S

  @classmethod
  def random(cls, B, S):
    X = np.random.random((GRID_SIZE, GRID_SIZE)) > random.random()
    return cls(X, B, S)

  @classmethod
  def empty(cls, B, S):
    X = np.empty((GRID_SIZE, GRID_SIZE), dtype=bool)
    return cls(X, B, S)

  def step(self, steps=1):
    K = np.ones((3, 3))
    cache = self.X.copy()
    for _ in range(steps):
      n = convolve2d(self.X, K, mode='same', boundary='wrap') - self.X
      res = (~self.X & np.isin(n, self.B)) | (self.X & np.isin(n, self.S))
      if (cache == res).all():
        self.X = res
        return False
      else:
        cache = res
    self.X = res
    return True

class MimicCA(CA):
  def step_from(self, X, steps=1):
    self.X = X
    return self.step(steps)
