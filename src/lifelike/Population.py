import math
import random
import numpy as np

from lifelike.CAs import MimicCA
from lifelike.Rulestring import Rulestring
from lifelike.constants import CHROMOSOME_LEN


class Population:
  def __init__(self, pop_size, elitism, mutation, trueB, trueS, ics, init_method, hyperparams):
    self.pop_size = pop_size
    self.elitism = elitism
    self.mutation = mutation
    self.elite_n = math.floor(elitism * pop_size)
    self.child_n = pop_size - self.elite_n
    self.trueB = trueB
    self.trueS = trueS
    self.ics = ics
    self.hyperparams = hyperparams
    if init_method == 'binary':
      self.inds = np.array([Rulestring.random_binary() for _ in range(self.elite_n)])
    elif init_method == 'decimal':
      self.inds = np.array([Rulestring.random_decimal() for _ in range(self.elite_n)])
    else:
      raise Exception("Unsupported init_method for Population")
    self.visited = set(self.inds)

  def iterate(self):
    self.crossover()
    self.mutate()
    loss = self.loss()
    self.visited.update(self.inds)
    self.update(loss)
    return self.evaluate(loss)

  def update(self, loss):
    self.inds = self.inds[loss.argsort()]
    self.inds = self.inds[:self.elite_n]

  def evaluate(self, loss):
    return 1 - np.mean(np.sort(loss)[:self.elite_n])

  def crossover(self):
    children = []
    for _ in range(self.child_n):
      cpoint = random.randint(1, CHROMOSOME_LEN - 1)
      parents = np.random.choice(self.inds, 2, replace=False)
      a, b = parents[0], parents[1]
      left = a.get_rstring()[:cpoint]
      right = b.get_rstring()[cpoint:]
      child = Rulestring.from_rstring(int(left + right, 2))
      children.append(child)
    self.inds = np.append(self.inds, np.array(children))

  def mutate(self):
    for ind in self.inds[self.elite_n:]:
      ind.mutate(self.mutation)

  def loss(self):
    true = MimicCA.empty(self.trueB, self.trueS)
    return np.array([r.loss(true, self.ics, self.hyperparams) for r in self.inds])

  def goal_found(self):
    return set(self.trueB) == set(self.inds[0].b) and set(self.trueS) == set(self.inds[0].s)

  def num_unique_inds(self):
    return len(set(i.rstring for i in self.inds))

  def __str__(self):
    res = "["
    for i in self.inds:
      res += f"{i.get_rstring()}, "
    res += "]"
    return res
