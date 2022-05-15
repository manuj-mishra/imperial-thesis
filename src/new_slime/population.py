import math
import random
import numpy as np

from new_slime.chromosome import Chromosome

EVAL_ITER = 5

class Population:
  def __init__(self, size_bias, pop_size, elitism, mutation):
    self.inds = np.array([Chromosome() for _ in range(pop_size)])
    self.pop_size = pop_size
    self.size_bias = size_bias
    self.elitism = elitism
    self.mutation = mutation
    self.elite_n = math.floor(elitism * pop_size)
    self.child_n = pop_size - self.elite_n

  def iterate(self):
    # Selection
    mean_food, mean_size, _ = self.select()

    # Crossover
    self.crossover()

    # Mutate
    self.mutate(self.elite_n // 5)
    return mean_food, mean_size

  def select(self):
    scores, mean_food, mean_size = self.evaluate()
    sorted_scores = np.sort(scores)[::-1]
    self.inds = self.inds[(-scores).argsort()]
    self.inds = self.inds[:self.elite_n]
    return mean_food, mean_size, sorted_scores[:self.elite_n]

  def crossover(self):
    children = []
    for _ in range(self.child_n):
      parents = np.random.choice(self.inds, 2, replace=False)
      p, q = parents[0], parents[1]
      cpoint = random.randint(1, 8)
      birth = p.b[:cpoint] + q.b[cpoint:]
      cpoint = random.randint(1, 8)
      survival = p.s[:cpoint] + q.s[cpoint:]
      child = Chromosome(birth, survival)
      children.append(child)
    self.inds = np.append(self.inds, np.array(children))

  def mutate(self, non_mutate_n):
    for ind in self.inds[non_mutate_n:]:
      ind.mutate_chrom(self.mutation)

  def evaluate(self):
    foods = []
    sizes = []
    for r in self.inds:
      food, size = r.evaluate(n_iters=EVAL_ITER)
      foods.append(food)
      sizes.append(size)

    foods = np.array(foods)
    sizes = np.array(sizes)

    n = len(self.inds)
    food_ixs = np.argsort(foods)
    size_ixs = np.argsort(sizes)

    food_ranks = np.empty(n)
    food_ranks[food_ixs] = np.linspace(0, 1, num=n)
    size_ranks = np.empty(n)
    size_ranks[size_ixs] = np.linspace(0, 1, num=n)
    scores = ((1 - self.size_bias) * food_ranks) + (self.size_bias * size_ranks)
    return scores, np.mean(foods), np.mean(sizes)

  def __str__(self):
    res = "["
    for i in self.inds:
      res += f"{i.get_rstring()}, "
    res += "]"
    return res