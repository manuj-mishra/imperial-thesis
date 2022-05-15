import math
import random
import numpy as np

from gray_scott.CAs import MimicCA
from gray_scott.Chromosome import Chromosome

EVAL_ITERS = 1  # Number of CAs simulated per rulestring
EVAL_STEPS = 20  # Number of steps evaluated per CA
MAX_STEP_SIZE = 200  # Max size of a single CA step


class Population:
  def __init__(self, pop_size, elitism, alpha, true_f, true_k):
    self.inds = np.array([Chromosome.near_threshold() for _ in range(pop_size)])
    self.pop_size = pop_size
    self.elitism = elitism
    self.alpha = alpha
    self.elite_n = math.floor(elitism * pop_size)
    self.child_n = pop_size - self.elite_n
    self.true_f = true_f
    self.true_k = true_k

  def iterate(self):
    self.crossover()
    self.mutate()
    loss = self.loss()
    self.update(loss)
    return self.evaluate(loss)

  def update(self, loss):
    self.inds = self.inds[loss.argsort()]
    self.inds = self.inds[:self.elite_n]

  def evaluate(self, loss):
    return np.mean(np.sort(loss)[:self.elite_n])

  def crossover(self):
    # BLX-alpha crossover
    children = []
    for _ in range(self.child_n):
      parents = np.random.choice(self.inds, 2, replace=False)
      a, b = parents[0], parents[1]
      deltF = abs(a.f - b.f)
      deltK = abs(a.k - b.k)
      child = Chromosome(
        f=a.f if random.random() < 0.5 else b.f,
        k=a.k if random.random() < 0.5 else b.k,
        df=np.random.uniform(low=max(min(a.df, b.df) - self.alpha * deltF, 0), high=max(a.df, b.df) + self.alpha * deltF),
        dk=np.random.uniform(low=max(min(a.dk, b.dk) - self.alpha * deltK, 0), high=max(a.dk, b.dk) + self.alpha * deltK)
      )
      children.append(child)
    self.inds = np.append(self.inds, np.array(children))

  def mutate(self):
    non_mutate_n = self.elite_n // 5
    for ind in self.inds[non_mutate_n:]:
      ind.mutate()

  def loss(self):
    true = MimicCA.empty(self.true_f, self.true_k)
    return np.array([r.loss(EVAL_ITERS, EVAL_STEPS, MAX_STEP_SIZE, true) for r in self.inds])

  def goal_found(self):
    pass

  def num_unique_inds(self):
    pass

  def __str__(self):
    pass
