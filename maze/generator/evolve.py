from generator.generate_maze import generate_maze
from generator.region_merge import get_regions, region_merge


class Population:
  def __init__(self):
    self.inds = []

class CA:
  def __init__(self, B, S):
    X = generate_maze(B, S, folder='temp/gen_frames')
    cells, regions, M = get_regions(X, folder='temp/reg_frames')
    M = region_merge(regions, cells, M, folder='temp/merge_frames')
    self.maze =

def run_experiment(offspring_ratio=1, elitism=0.5, crossover=1, mutation=0.05, epochs=100):
  population = init_pop()
