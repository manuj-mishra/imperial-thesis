import csv
import math
import random
import time
from maze.rulestring import Rulestring
import numpy as np

class Population:
  def __init__(self, pop_size, fit_ratio, elitism, mutation):
    self.inds = np.array([Rulestring() for _ in range(pop_size)])
    self.fit_ratio = fit_ratio
    self.elite_n = math.floor(elitism * pop_size)
    self.child_n = pop_size - self.elite_n
    self.mutation = mutation

  def iterate(self):
    # Selection
    avg, max, fail, _ = self.select()

    # Crossover
    self.crossover()

    # Mutate
    self.mutate(0)

    return avg, max, fail

  def select(self):
    ranks, fails = self.evaluate()
    sorted_ranks = np.sort(ranks)[::-1]
    avg_fitness = np.mean(sorted_ranks[:self.elite_n])
    max_fitness = np.max(ranks)
    self.inds = self.inds[(-ranks).argsort()]
    self.inds = self.inds[:self.elite_n]
    return avg_fitness, max_fitness, fails, sorted_ranks[:self.elite_n]

  def crossover(self):
    children = []
    for _ in range(self.child_n):
      cpoint = random.randint(1, 15)
      parents = np.random.choice(self.inds, 2, replace=False)
      a, b = parents[0], parents[1]
      left = a.get_rstring()[:cpoint]
      right = b.get_rstring()[cpoint:]
      child = Rulestring(int(left + right, 2))
      children.append(child)
    self.inds = np.append(self.inds, np.array(children))

  def mutate(self, elite_n):
    for rstring in self.inds[elite_n:]:
      rstring.mutate(self.mutation)

  def evaluate(self):
    scores = []
    for r in self.inds:
      sol_len, n_ends = r.evaluate(n_iters=5)
      scores.append(self.fit_ratio * sol_len + (1 - self.fit_ratio) * n_ends)
    scores = np.array(scores)
    fails = np.count_nonzero(scores == 0)
    return scores, fails

  def __str__(self):
    res = "["
    for i in self.inds:
      res += f"{i.get_rstring()}, "
    res += "]"
    return res



