import csv
import time

import matplotlib.pyplot as plt
import numpy as np

from maze.maze_ca import MazeCA
from maze.population import Population

EPOCH_N = 100

def run_experiment(path_len_bias, pop_size=100, elitism=0.5, mutation=0.05, novelty=1, epoch_n=EPOCH_N):
  start_time = time.time()
  mean_deads = []
  mean_paths = []
  pop = Population(pop_size, path_len_bias, elitism, mutation, novelty)
  for epoch in range(epoch_n - 1):
    print("Epoch:", epoch + 1)
    mean_dead_ends, mean_path_lens = pop.iterate()
    mean_deads.append(mean_dead_ends)
    mean_paths.append(mean_path_lens)

  print("Epoch:", epoch_n)
  mean_dead_ends, mean_path_lens, elite_scores = pop.select()
  mean_deads.append(mean_dead_ends)
  mean_paths.append(mean_path_lens)
  print(f"Training Time: {((time.time() - start_time) / epoch_n):.2f} s per epoch")
  save_experiments(pop, elite_scores, mean_paths, mean_deads)

  return np.mean(mean_deads[-1 * (EPOCH_N//10):]), np.mean(mean_paths[-1 * (EPOCH_N//10):])


def save_experiments(pop, elite_scores, mean_paths, mean_deads):
  exp_n = f"bias{pop.path_len_bias * 100}_pop{pop.pop_size}_el{pop.elitism * 100}_mut{pop.mutation * 100}"
  for rulestring in pop.inds[:3]:
    rulestring.b.sort()
    rulestring.s.sort()

    rname = ''.join(str(i) for i in rulestring.b) + '_' + ''.join(str(i) for i in rulestring.s)
    rname = f"{exp_n}/{rname}"

    ca = MazeCA(rulestring.b, rulestring.s)
    ca.save_experiment(rname)

  dir = f'out/{exp_n}'
  file = open(f'{dir}/elite.csv', 'w+', newline='')
  with file:
    write = csv.writer(file)
    write.writerow([f"{e.b}_{e.s}" for e in pop.inds])
    write.writerow([e.get_rstring() for e in pop.inds])
    write.writerow(elite_scores)

  epochs = [i for i in range(1, EPOCH_N + 1)]
  fig, ax1 = plt.subplots()

  color = 'tab:red'
  ax1.set_xlabel('Epoch')
  ax1.set_ylabel('# of dead ends')
  ax1.plot(epochs, mean_deads, color=color)
  ax1.tick_params(axis='y', labelcolor=color)

  color = 'tab:green'
  ax2 = ax1.twinx()
  ax2.set_ylabel('Length of solution path')
  ax2.plot(epochs, mean_paths, color=color)
  ax2.tick_params(axis='y', labelcolor=color)

  fig.tight_layout()
  plt.savefig(f'{dir}/fitness.png', bbox_inches='tight')


if __name__ == "__main__":
  run_experiment(0.5, novelty=1)
  # N = 3
  # mean_dead = []
  # mean_path = []
  # for _ in range(N):
  #   d, p = run_experiment(0.5, novelty=1)
  #   mean_dead.append(d)
  #   mean_path.append(p)
  #
  # for _ in range(N):
  #   d, p = run_experiment(0.5, novelty=3)
  #   mean_dead.append(d)
  #   mean_path.append(p)
  #
  # fig, ax1 = plt.subplots()
  #
  # c = ['r' for _ in range(N)] + ['g' for _ in range(N)]
  # ax1.scatter(mean_dead, mean_path, c=c)
  #
  # fig.tight_layout()
  # plt.savefig(f'out/diversity.png', bbox_inches='tight')
