import math
import random
import numpy as np
import pandas as pd

from novelty.CAs import MimicCA
from novelty.Rulestring import Rulestring
from novelty.constants import CHROMOSOME_LEN


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
    poploss = self.loss()
    self.update(poploss)

  def iterate(self):
    self.crossover()
    self.mutate()
    poploss = self.loss()
    self.visited.update(self.inds)
    self.update(poploss)
    return self.evaluate(poploss)

  def update(self, poploss):
    self.inds = self.inds[poploss.argsort()]
    self.inds = self.inds[:self.elite_n]

  def evaluate(self, poploss):
    return 1 - np.mean(np.sort(poploss)[:self.elite_n])

  def crossover(self):
    children = []
    while len(children) < self.child_n:
      cpoint = random.randint(1, CHROMOSOME_LEN - 1)
      parents = np.random.choice(self.inds, 2, replace=False)
      a, b = parents[0].get_rstring(), parents[1].get_rstring()
      left_a, right_a = a[:cpoint], a[cpoint:]
      left_b, right_b = b[:cpoint], b[cpoint:]
      child1 = Rulestring.from_rstring(int(left_a + right_b, 2))
      if child1.isvalid():
        children.append(child1)
      child2 = Rulestring.from_rstring(int(left_b + right_a, 2))
      if child1.isvalid():
        children.append(child2)
    self.inds = np.append(self.inds, np.array(children))

  def mutate(self):
    for ind in self.inds:
      ind.mutate(self.mutation)
      while not ind.isvalid():
        ind.mutate(self.mutation)

  def loss(self):
    true = MimicCA.empty(self.trueB, self.trueS)
    for r in self.inds:
      r.calc_loss(true, self.ics, self.hyperparams)
    avg_nov = np.mean([i.novelty for i in self.inds])
    poploss = [0.8 * abs(r.novelty - avg_nov) + 0.2 * r.loss for r in self.inds]
    return np.array(poploss)

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
