import csv
import time

import matplotlib.pyplot as plt
import numpy as np

from slime.old_slime.population import Population
from slime.old_slime.slime_ca import SlimeCA

EPOCH_N = 20


def run_experiment(food_eaten_bias, pop_size=50, elitism=0.3, mutation=0.25, epoch_n=EPOCH_N):
  start_time = time.time()
  mean_foods = []
  mean_sizes = []
  pop = Population(food_eaten_bias, pop_size, elitism, mutation, epoch_n)
  for epoch in range(epoch_n - 1):
    print("Epoch:", epoch + 1)
    mean_food_eaten, mean_slime_size = pop.iterate(epoch)
    mean_foods.append(mean_food_eaten)
    mean_sizes.append(mean_slime_size)

  print("Epoch:", epoch_n)
  mean_food_eaten, mean_slime_size, elite_scores = pop.select()
  mean_foods.append(mean_food_eaten)
  mean_sizes.append(mean_slime_size)

  print(f"Training Time: {((time.time() - start_time) / epoch_n):.2f} s per epoch")
  save_experiments(pop, elite_scores, mean_foods, mean_sizes)
  return np.mean(mean_foods[-3:]), np.mean(mean_sizes[-3:])


def save_experiments(pop, elite_scores, mean_foods, mean_sizes):
  exp_n = f"bias{pop.food_eaten_bias * 100}_pop{pop.pop_size}_el{pop.elitism * 100}"
  for chromosome in pop.inds[:3]:
    rname = f"{exp_n}/{chromosome.id()}"

    ca = SlimeCA(chromosome)
    ca.save_experiment(rname)

  dir = f'out/{exp_n}'
  file = open(f'{dir}/elite.csv', 'w+', newline='')
  with file:
    write = csv.writer(file)
    write.writerow([e.coefs for e in pop.inds])
    write.writerow(elite_scores)

  epochs = [i for i in range(1, EPOCH_N + 1)]
  fig, ax1 = plt.subplots()

  color = 'tab:green'
  ax1.set_xlabel('Epochs')
  ax1.set_ylabel('Slimes touching food')
  ax1.plot(epochs, mean_foods, color='green')
  ax1.tick_params(axis='y', labelcolor=color)

  color = 'tab:red'
  ax2 = ax1.twinx()
  ax2.set_ylabel('Size of slime')
  ax2.plot(epochs, mean_sizes, color='red')
  ax2.tick_params(axis='y', labelcolor=color)

  fig.tight_layout()
  plt.savefig(f'{dir}/fitness.png', bbox_inches='tight')


if __name__ == "__main__":
  run_experiment(0.49)
  # mean_foods = []
  # mean_sizes = []
  # biases = np.linspace(0.45, 0.49, num=9)
  # for bias in biases:
  #   f, s = run_experiment(bias)
  #   mean_foods.append(f)
  #   mean_sizes.append(s)
  #
  # fig, ax1 = plt.subplots()
  #
  # color = 'tab:green'
  # ax1.set_xlabel('Bias towards food')
  # ax1.set_ylabel('Slimes touching food')
  # ax1.plot(biases, mean_foods, color=color)
  # ax1.tick_params(axis='y', labelcolor=color)
  #
  # color = 'tab:red'
  # ax2 = ax1.twinx()
  # ax2.set_ylabel('Size of slime')
  # ax2.plot(biases, mean_sizes, color=color)
  # ax2.tick_params(axis='y', labelcolor=color)
  #
  # fig.tight_layout()
  # plt.savefig(f'out/fine_bias_tuning.png', bbox_inches='tight')