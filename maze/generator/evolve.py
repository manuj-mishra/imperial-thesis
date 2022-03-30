import math
import random
from generator.rulestring import Rulestring
import numpy as np


class Population:
  def __init__(self, size, elitism=0.5, mutation=0.05):
    self.inds = np.array([Rulestring() for _ in range(size)])
    self.elitism = elitism
    self.mutation = mutation

  def iterate(self):
    elite_n = math.floor(self.elitism * len(self.inds))
    child_n = len(self.inds) - elite_n

    # Elitism
    print("Elitism")
    ranks = self.evaluate()
    self.inds = self.inds[(-ranks).argsort()]
    self.inds = self.inds[:elite_n]

    # Crossover
    print("Crossover")
    self.inds = np.append(self.inds, self.crossover(child_n), 0)

    # Mutate
    print("Mutation")
    self.mutate()

  def crossover(self, child_n):
    children = []
    for _ in range(child_n):
      cpoint = random.randint(1, 15)
      a, b = random.sample(self.inds, 2)
      left = a.get_rstring()[:cpoint]
      right = b.get_rstring()[cpoint:]
      child = int(left + right, 2)
      children.append(child)
    return np.array(children)

  def mutate(self):
    for rstring in self.inds:
      rstring.mutate(self.mutation)

  def evaluate(self):
    scores = np.array([r.evaluate() for r in self.inds])
    print("Num fails:", np.count_nonzero(scores == 0))
    return scores

  def __str__(self):
    res = "["
    for i in self.inds:
      res += f"{i.get_rstring()}, "
    res += "]"
    return res


if __name__ == "__main__":
  pop = Population(size=10)
  print(pop)
  pop.iterate()
  print(pop)
