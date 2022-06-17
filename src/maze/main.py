import csv
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame

from maze.maze_ca import MazeCA
from maze.population import Population

EPOCH_N = 10

def run_experiment(path_len_bias, pop_size=50, elitism=0.2, mutation=0.0, epoch_n=EPOCH_N):
  start_time = time.time()
  mean_deads = []
  mean_paths = []
  pop = Population(pop_size, path_len_bias, elitism, mutation)
  for epoch in range(epoch_n - 1):
    # print("Epoch:", epoch + 1)
    mean_dead_ends, mean_path_lens = pop.iterate()
    mean_deads.append(mean_dead_ends)
    mean_paths.append(mean_path_lens)

  mean_dead_ends, mean_path_lens = pop.select()
  mean_deads.append(mean_dead_ends)
  mean_paths.append(mean_path_lens)
  # save_experiments(pop, mean_paths, mean_deads)
  return mean_dead_ends, mean_path_lens


def save_experiments(pop, mean_paths, mean_deads):
  exp_n = f"pop{pop.pop_size}_el{pop.elitism * 100}_mut{pop.mutation * 100}"
  for rulestring in pop.inds[:(pop.elite_n//5)]:
    rulestring.b.sort()
    rulestring.s.sort()

    rname = ''.join(str(i) for i in rulestring.b) + '_' + ''.join(str(i) for i in rulestring.s)
    rname = f"{exp_n}/{rname}"

    ca = MazeCA(rulestring.b, rulestring.s)
    ca.save_experiment(rname)

  dir = f'out/{exp_n}'
  os.makedirs(dir, exist_ok=True)
  file = open(f'{dir}/elite.csv', 'w+', newline='')
  with file:
    write = csv.writer(file)
    write.writerow([f"{e.b}_{e.s}" for e in pop.inds])
    write.writerow([e.get_rstring() for e in pop.inds])
    # write.writerow(elite_scores)

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
  # ABLATION
  # d = []
  # p = []
  # n = 10
  # for _ in range(n):
  #   md, mp = run_experiment(0.5)
  #   d.append(md)
  #   p.append(mp)
  #
  # df = DataFrame.from_dict({"category": ["no mutation or crossover"]*n, "d":d , "p": p})
  # df.to_csv("maze-nothing.csv", index=False)

  # SELECTION TYPE EXPERIMENT
  d = []
  p = []
  for i in range(30):
    print("Exp", i)
    md, mp = run_experiment(0.5)
    d.append(md)
    p.append(mp)

  df = DataFrame.from_dict({"type": ["Relative Roulette"] * len(d), "d": d, "p": p})
  df.to_csv("roulette_ranks.csv")

  # BIAS EXPERIMENT
  # b = []
  # d = []
  # p = []
  # for bias in np.linspace(0.0, 100.0, num =11):
  #   print(bias)
  #   md, mp = run_experiment(bias)
  #   b.append(bias)
  #   d.append(md)
  #   p.append(mp)
  #
  # df = DataFrame.from_dict({"b": b, "d": d, "p": p})
  # df.to_csv("bias_tuning_hyp.csv")

  # # HYPERPARAM EXPERIMENT:
  # mean_deads = []
  # mean_paths = []
  # train_times = []
  # ids = []
  # for pop in (20, 50, 100):
  #   for el in (0.1, 0.2, 0.5):
  #     for mut in (0.01, 0.05, 0.1):
  #       for ep in (10, 50, 100):
  #         ids.append(f"pop{pop}_el{int(el * 100)}_mut{int(mut * 100)}_ep{int(ep*100)}")
  #         mean_dead_ends, mean_path_lens, train_time = run_experiment(0.5, pop_size=pop, elitism=el, mutation=mut, epoch_n=ep)
  #         mean_deads.append(mean_dead_ends)
  #         mean_paths.append(mean_path_lens)
  #         train_times.append(train_time)
  # df = DataFrame.from_dict({"id": ids, "d": mean_deads, "p":mean_paths, "t": train_times})
  # df.to_csv("maze_hyperparam.csv")

  # # NOVELTY EXPERIMENT
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
