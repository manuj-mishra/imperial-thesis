import math
import random

import numpy as np

from maze.rulestring import Rulestring

EVAL_ITER = 5

class Population:
  def __init__(self, pop_size, path_len_bias, elitism, mutation, novelty):
    self.inds = np.array([Rulestring() for _ in range(pop_size)])
    self.pop_size = pop_size
    self.path_len_bias = path_len_bias
    self.elitism = elitism
    self.mutation = mutation
    self.novelty = novelty
    self.elite_n = math.floor(elitism * pop_size)
    self.child_n = pop_size - self.elite_n

  def iterate(self):
    # Selection
    mean_dead_ends, mean_path_lens, _ = self.select()

    # Crossover
    self.crossover()

    # Mutate
    self.mutate(self.elite_n // 5, self.novelty)
    # best_diversity = 0
    # best_mutation = self.inds.copy()
    # for _ in range(self.novelty):
    #   self.mutate(self.elite_n // 5)
    #   d = self.diversity()
    #   if d > best_diversity:
    #     best_diversity = d
    #     best_mutation = self.inds.copy()
    # self.inds = best_mutation

    return mean_dead_ends, mean_path_lens

  def select(self):
    scores, mean_dead_ends, mean_path_lens = self.evaluate()
    sorted_scores = np.sort(scores)[::-1]
    self.inds = self.inds[(-scores).argsort()]
    self.inds = self.inds[:self.elite_n]
    return mean_dead_ends, mean_path_lens, sorted_scores[:self.elite_n]

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

  def mutate(self, non_mutate_n, diversity_n):
    if diversity_n == 1:
      for ind in self.inds[non_mutate_n:]:
        ind.mutate(self.mutation)
      return

    for ind in self.inds[non_mutate_n:]:
      original = Rulestring(ind.rstring)
      diversity_score = 0
      for _ in range(diversity_n):
        rstring = original.mutate(self.mutation)
        d = self.diversity_to(rstring)
        if d > diversity_score:
          diversity_score = d
          ind.set_rstring(rstring)

  def evaluate(self):
    dead_ends = []
    path_lens = []
    for r in self.inds:
      dead_end, path_len, reachable = r.evaluate(n_iters=EVAL_ITER)
      dead_ends.append(dead_end)
      path_lens.append(path_len)

    dead_ends = np.array(dead_ends)
    path_lens = np.array(path_lens)

    n = len(self.inds)
    dead_ixs = np.argsort(dead_ends)
    path_ixs = np.argsort(path_lens)

    dead_ranks = np.empty(n)
    dead_ranks[dead_ixs] = np.linspace(0, 1, num=n)
    path_ranks = np.empty(n)
    path_ranks[path_ixs] = np.linspace(0, 1, num=n)
    scores = ((1 - self.path_len_bias) * dead_ranks) + (self.path_len_bias * path_ranks)
    scores = np.where(path_lens == 0, 0, scores)
    return scores, np.mean(dead_ends[dead_ends != 0]), np.mean(path_lens[path_lens != 0])

  def diversity_to(self, rstring):
    return np.mean([hamming(rstring, i.rstring) for i in self.inds])

  def diversity(self):
    return np.mean([hamming(i.rstring, j.rstring) for i in self.inds for j in self.inds])

  def __str__(self):
    res = "["
    for i in self.inds:
      res += f"{i.get_rstring()}, "
    res += "]"
    return res

def hamming(a, b):
  # Returns hamming distance between a and b
  c = a ^ b
  count = 0
  while c:
    c &= c - 1
    count += 1
  return count
