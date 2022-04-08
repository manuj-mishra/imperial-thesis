import math
from slime.chromosome import Chromosome
import numpy as np

class Population:
  def __init__(self, pop_size, fit_ratio, elitism, mutation):
    self.inds = np.array([Chromosome() for _ in range(pop_size)])
    self.fit_ratio = fit_ratio
    self.elite_n = math.floor(elitism * pop_size)
    self.child_n = pop_size - self.elite_n
    self.mutation = mutation

  def iterate(self):
    # Selection
    avg, max, _ = self.select()

    # Crossover
    self.blend_crossover(0.5)

    # Mutate
    self.mutate(0)

    return avg, max

  def select(self):
    ranks = self.evaluate()
    sorted_ranks = np.sort(ranks)[::-1]
    avg_fitness = np.mean(sorted_ranks[:self.elite_n])
    max_fitness = np.max(ranks)
    self.inds = self.inds[(-ranks).argsort()]
    self.inds = self.inds[:self.elite_n]
    return avg_fitness, max_fitness, sorted_ranks[:self.elite_n]

  def blend_crossover(self, alpha):
    children = []
    for _ in range(self.child_n):
      parents = np.random.choice(self.inds, 2, replace=False)
      a, b = parents[0], parents[1]
      child = Chromosome()
      for k in a.b_coefs:
        gamma = np.random.uniform(-1 * alpha, 1 + alpha)
        child.b_coefs[k] = gamma * a.b_coefs[k] + (1 - gamma) * b.b_coefs[k]
      for k in a.s_coefs:
        gamma = np.random.uniform(-1 * alpha, 1 + alpha)
        child.s_coefs[k] = gamma * a.s_coefs[k] + (1 - gamma) * b.s_coefs[k]
      children.append(child)
    self.inds = np.append(self.inds, np.array(children))

  def mutate(self, elite_n):
    for chrom in self.inds[elite_n:]:
      chrom.gaussian_mutation(self.mutation)

  def evaluate(self):
    scores = []
    for r in self.inds:
      food_eaten, slime_size = r.evaluate(n_iters=5)
      scores.append(self.fit_ratio * food_eaten - (1 - self.fit_ratio) * slime_size)
    scores = np.array(scores)
    return scores

  def __str__(self):
    res = "["
    for i in self.inds:
      res += f"{str(i.coefs)}, "
    res += "]"
    return res

if __name__ == "__main__":
  p = Population(pop_size=50, fit_ratio=0.9, elitism=0.2, mutation=0.2)