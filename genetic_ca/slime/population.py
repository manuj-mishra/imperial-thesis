import math
from slime.chromosome import Chromosome
import numpy as np

EVAL_ITER = 5


class Population:
  def __init__(self, food_eaten_bias, pop_size, elitism, mutation, max_epoch):
    self.inds = np.array([Chromosome() for _ in range(pop_size)])
    self.food_eaten_bias = food_eaten_bias
    self.pop_size = pop_size
    self.elitism = elitism
    self.mutation = mutation
    self.max_epoch = max_epoch
    self.elite_n = math.floor(elitism * pop_size)
    self.child_n = pop_size - self.elite_n

  def iterate(self, curr_epoch):
    # Selection
    mean_food_eaten, mean_slime_size, _ = self.select()

    # Crossover
    self.blend_crossover(0.5)

    # Mutate
    self.mutate(self.elite_n, curr_epoch)

    return mean_food_eaten, mean_slime_size

  def select(self):
    scores, mean_food_eaten, mean_slime_size = self.evaluate()
    sorted_scores = np.sort(scores)[::-1]
    self.inds = self.inds[(-scores).argsort()]
    self.inds = self.inds[:self.elite_n]
    return mean_food_eaten, mean_slime_size, sorted_scores[:self.elite_n]

  def blend_crossover(self, alpha):
    children = []
    for _ in range(self.child_n):
      parents = np.random.choice(self.inds, 2, replace=False)
      a, b = parents[0], parents[1]
      child = Chromosome()
      for k in a.coefs:
        gamma = np.random.uniform(-1 * alpha, 1 + alpha)
        child.coefs[k] = gamma * a.coefs[k] + (1 - gamma) * b.coefs[k]
      children.append(child)
    self.inds = np.append(self.inds, np.array(children))

  def mutate(self, elite_n, curr_epoch):
    for chrom in self.inds[elite_n:]:
      chrom.single_non_uniform_mutation(curr_epoch, self.max_epoch)

  def evaluate(self):
    food_eaten = []
    slime_size = []
    for r in self.inds:
      fe, ss = r.evaluate(n_iters=EVAL_ITER)
      food_eaten.append(fe)
      slime_size.append(ss)

    food_eaten = np.array(food_eaten)
    slime_size = np.array(slime_size)

    n = len(self.inds)
    food_ixs = np.argsort(food_eaten)
    size_ixs = np.argsort((-slime_size))

    food_ranks = np.empty(n)
    food_ranks[food_ixs] = np.linspace(0, 1, num=n)
    size_ranks = np.empty(n)
    size_ranks[size_ixs] = np.linspace(0, 1, num=n)
    scores = ((1 - self.food_eaten_bias) * size_ranks) + (self.food_eaten_bias * food_ranks)
    # TODO: Is manual replacement below needed?
    # scores = np.where(slime_size == 0, 0, scores)
    return scores, np.mean(food_eaten), np.mean(slime_size)

  def __str__(self):
    res = "["
    for i in self.inds:
      res += f"{str(i.coefs)}, "
    res += "]"
    return res
