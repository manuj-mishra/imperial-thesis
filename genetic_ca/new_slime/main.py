import csv
import time

import matplotlib.pyplot as plt
import numpy as np

from new_slime.population import Population
from new_slime.slime_ca import SlimeCA

EPOCH_N = 50

def run_experiment(size_bias, pop_size=30, elitism=0.5, mutation=0.05, epoch_n=EPOCH_N):
  start_time = time.time()
  mean_foods = []
  mean_sizes = []
  pop = Population(size_bias, pop_size, elitism, mutation)
  for epoch in range(epoch_n - 1):
    print("Epoch:", epoch + 1)
    food, size = pop.iterate()
    mean_foods.append(food)
    mean_sizes.append(size)

  print("Epoch:", epoch_n)
  food, size, elite_scores = pop.select()
  mean_foods.append(food)
  mean_sizes.append(size)
  print(f"Training Time: {((time.time() - start_time) / epoch_n):.2f} s per epoch")
  # save_experiments(pop, elite_scores, mean_foods, mean_sizes)
  return np.mean(mean_foods[-1 * (EPOCH_N//10):]), np.mean(mean_sizes[-1 * (EPOCH_N//10):])


def save_experiments(pop, elite_scores, mean_food, mean_size):
  exp_n = f"bias{pop.size_bias * 100}_pop{pop.pop_size}_el{pop.elitism * 100}_mut{pop.mutation * 100}"
  for chromosome in pop.inds[:(pop.elite_n//5)]:
    rname = f"{exp_n}/{chromosome.id()}"
    ca = SlimeCA(chromosome.b, chromosome.s)
    ca.save_experiment(rname)

  dir = f'out/{exp_n}'
  file = open(f'{dir}/elite.csv', 'w+', newline='')
  with file:
    write = csv.writer(file)
    write.writerow([f"{e.id()}" for e in pop.inds])
    write.writerow(elite_scores)

  epochs = [i for i in range(1, EPOCH_N + 1)]
  fig, ax1 = plt.subplots()

  color = 'tab:red'
  ax1.set_xlabel('Epoch')
  ax1.set_ylabel('Food ratio')
  ax1.plot(epochs, mean_food, color=color)
  ax1.tick_params(axis='y', labelcolor=color)

  color = 'tab:green'
  ax2 = ax1.twinx()
  ax2.set_ylabel('Size ratio')
  ax2.plot(epochs, mean_size, color=color)
  ax2.tick_params(axis='y', labelcolor=color)

  fig.tight_layout()
  plt.savefig(f'{dir}/fitness.png', bbox_inches='tight')


if __name__ == "__main__":
  mean_foods = []
  mean_sizes = []
  biases = np.linspace(0.0, 1.0, num=6)
  for bias in biases:
    f, s = run_experiment(bias)
    mean_foods.append(f)
    mean_sizes.append(s)

  fig, ax1 = plt.subplots()

  color = 'tab:green'
  ax1.set_xlabel('Bias towards size')
  ax1.set_ylabel('Food ratio')
  ax1.plot(biases, mean_foods, color=color)
  ax1.tick_params(axis='y', labelcolor=color)

  color = 'tab:red'
  ax2 = ax1.twinx()
  ax2.set_ylabel('Size ratio')
  ax2.plot(biases, mean_sizes, color=color)
  ax2.tick_params(axis='y', labelcolor=color)

  fig.tight_layout()
  plt.savefig(f'out/bias_tuning.png', bbox_inches='tight')
