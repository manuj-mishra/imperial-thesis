import math
import random
import numpy as np

from lifelike.CAs import MimicCA
from lifelike.Rulestring import Rulestring
from lifelike.constants import CHROMOSOME_LEN


class Population:
  def __init__(self, pop_size, elitism, mutation, trueB, trueS, ics, init_method = 'binary', hyperparams=None):
    if hyperparams is None:
      hyperparams = {"max_step": 5, "eval_step": 10}
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
      self.inds = np.array([Rulestring.random_binary() for _ in range(pop_size)])
    elif init_method == 'decimal':
      self.inds = np.array([Rulestring.random_decimal() for _ in range(pop_size)])
    else:
      raise Exception("Unsupported init_method for Population")
    self.visited = set()
    loss = self.loss()
    self.update(loss)

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
    for _ in range(self.child_n // 2):
      cpoint = random.randint(1, CHROMOSOME_LEN - 1)
      parents = np.random.choice(self.inds, 2, replace=False)
      a, b = parents[0].get_rstring(), parents[1].get_rstring()
      left_a, right_a = a[:cpoint], a[cpoint:]
      left_b, right_b = b[:cpoint], b[cpoint:]
      child1 = Rulestring.from_rstring(int(left_a + right_b, 2))
      child2 = Rulestring.from_rstring(int(left_b + right_a, 2))
      children.append(child1)
      children.append(child2)
    self.inds = np.append(self.inds, np.array(children))

  def mutate(self):
    for ind in self.inds[self.elite_n//5:]:
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
