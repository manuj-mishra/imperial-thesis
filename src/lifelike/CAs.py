import random
import numpy as np
from scipy.signal import convolve2d

GRID_SIZE = 32


class CA:
  def __init__(self, X, B, S):
    self.X = X
    self.B = B
    self.S = S
    self.K = np.ones((3, 3))
    self.volatility = 0
    self.steps = 0

  @classmethod
  def random(cls, B, S):
    X = np.random.random((GRID_SIZE, GRID_SIZE)) < random.random()
    return cls(X, B, S)

  @classmethod
  def empty(cls, B, S):
    X = np.empty((GRID_SIZE, GRID_SIZE), dtype=bool)
    return cls(X, B, S)

  def step(self, steps=1):
    cache = self.X.copy()
    for _ in range(steps):
      n = convolve2d(self.X, self.K, mode='same', boundary='wrap') - self.X
      res = (~self.X & np.isin(n, list(self.B))) | (self.X & np.isin(n, list(self.S)))
      self.steps += 1
      if (cache == res).all():
        self.X = res
        return False
      else:
        cache = res
    self.X = res
    # print(np.sum(self.X))
    return True

  def simple_step(self):
    prior = self.X.copy()
    n = convolve2d(self.X, self.K, mode='same', boundary='wrap') - self.X
    self.X = (~self.X & np.isin(n, list(self.B))) | (self.X & np.isin(n, list(self.S)))
    self.volatility += (prior ^ self.X)
    self.steps += 1

class MimicCA(CA):
  def step_from(self, X, steps=1):
    self.X = X
    return self.step(steps)

if __name__ == "__main__":
  ca = CA.random({}, {})
  ca.step(10)