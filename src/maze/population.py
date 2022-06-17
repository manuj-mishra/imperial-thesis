import math
import random

import numpy as np
from pandas import DataFrame

from lifelike.constants import CHROMOSOME_LEN
from maze.rulestring import Rulestring

EVAL_ITER = 5

class Population:
  def __init__(self, pop_size, path_len_bias, elitism, mutation):
    self.pop_size = pop_size
    self.path_len_bias = path_len_bias
    self.elitism = elitism
    self.mutation = mutation
    self.elite_n = math.floor(elitism * pop_size)
    self.child_n = pop_size - self.elite_n
    self.inds = np.array([Rulestring() for _ in range(self.elite_n)])

  def iterate(self):
    # Selection
    mean_dead_ends, mean_path_lens = self.select()

    # Crossover
    self.crossover()

    # Mutate
    # self.mutate(self.elite_n // 5)
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
    # sorted_scores = np.sort(scores)[::-1]
    # self.inds = self.inds[(-scores).argsort()]
    # self.inds = self.inds[:self.elite_n]
    self.inds = random.choices(self.inds, scores, k=self.elite_n)
    return mean_dead_ends, mean_path_lens
    # return mean_dead_ends, mean_path_lens, sorted_scores[:self.elite_n]

  def crossover(self):
    children = []
    for _ in range(self.child_n // 2):
      cpoint = random.randint(1, CHROMOSOME_LEN - 1)
      parents = np.random.choice(self.inds, 2, replace=False)
      a, b = parents[0].get_rstring(), parents[1].get_rstring()
      left_a, right_a = a[:cpoint], a[cpoint:]
      left_b, right_b = b[:cpoint], b[cpoint:]
      child1 = Rulestring(int(left_a + right_b, 2))
      child2 = Rulestring(int(left_b + right_a, 2))
      children.append(child1)
      children.append(child2)
    self.inds = np.append(self.inds, np.array(children))


  def mutate(self, non_mutate_n):
    for ind in self.inds[non_mutate_n:]:
      ind.mutate(self.mutation)

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
    # scores1 = ((1 - self.path_len_bias) * dead_ixs) + (self.path_len_bias * path_ixs)
    scores2 = ((1 - self.path_len_bias) * dead_ranks) + (self.path_len_bias * path_ranks)
    # scores1 = np.where(path_lens == 0, 0, scores1)
    scores2 = np.where(path_lens == 0, 0, scores2)
    return scores2, np.mean(dead_ends[dead_ends != 0]), np.mean(path_lens[path_lens != 0])

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

# if __name__ == "__main__":
#   pop = Population(100, 0.5, 0.5, 0.05)
#   s1, s2, _, _= pop.evaluate()
#   obj = {k:None for k in range(30)}
#   rel = {k:None for k in range(30)}
#   obj[0] = s1
#   rel[0] = s2
#   for epoch in range(1, 31):
#     print("Epoch:", epoch)
#     _, _, s1, s2 = pop.iterate()
#     obj[epoch] = s1
#     rel[epoch] = s2
#   DataFrame.from_dict(obj).to_csv("obj.csv")
#   DataFrame.from_dict(rel).to_csv("rel.csv")


