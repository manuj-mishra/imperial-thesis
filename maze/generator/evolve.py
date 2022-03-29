from generator.generate_maze import generate_maze
from generator.region_merge import get_regions, region_merge
from generator.rulestring import Rulestring


class Population:
  def __init__(self, size):
    self.inds = [Rulestring() for _ in range(size)]

  def mutate(self, p):
    for rstring in self.inds:
      rstring.mutate(p)

  def evaluate(self):
    return [r.evaluate() for r in self.inds]

# def run_experiment(offspring_ratio=1, elitism=0.5, crossover=1, mutation=0.05, epochs=100):
#   population = init_pop()
