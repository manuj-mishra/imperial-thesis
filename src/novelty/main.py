import csv
import os
import time
import random

import numpy as np
from matplotlib import pyplot as plt
from pandas import DataFrame

from lifelike.CAs import GRID_SIZE
from lifelike.Population import Population
from lifelike.constants import CHROMOSOME_LEN
from alive_progress import alive_bar

# Repetition variables
EXPERIMENT_N = 10  # Experiments per density evaluated
ACCURACY_EPOCH_N = 30  # Epochs for accuracy experiment
CONV_MAX_EPOCH_N = 20  # Epochs for convergence experiment

# Training variables
POPULATION_SIZE = 100
ELITISM_RATE = 0.2
MUTATION_RATE = 0.05


def accuracy_experiment(trueB, trueS, pop_size=POPULATION_SIZE, elitism=ELITISM_RATE, mutation=MUTATION_RATE,
                        epoch_n=ACCURACY_EPOCH_N):
  pop = Population(pop_size, elitism, mutation, trueB, trueS)
  for _ in range(epoch_n - 1):
    pop.iterate()
  return pop.iterate()


def convergence_experiment(trueB, trueS, pop_size=POPULATION_SIZE, elitism=ELITISM_RATE, mutation=MUTATION_RATE,
                           epoch_n=CONV_MAX_EPOCH_N):
  pop = Population(pop_size, elitism, mutation, trueB, trueS)
  counter = 0
  for _ in range(epoch_n):
    if pop.goal_found():
      return counter
    pop.iterate()
    counter += 1
  return None


def test_single_GA(trueB, trueS, pop_size=POPULATION_SIZE, elitism=ELITISM_RATE, mutation=MUTATION_RATE,
                   epoch_n=ACCURACY_EPOCH_N):
  # with open('ics.npy', 'rb') as icfile:
  #   ics = np.load(icfile)[:100]
  ics = [np.random.random((GRID_SIZE, GRID_SIZE)) > random.random() for i in range(20)]
  # print(len(ics), "ics")
  pop = Population(pop_size, elitism, mutation, trueB, trueS, ics, init_method='decimal')
  accuracies = [pop.evaluate(pop.loss())]
  unique_inds = [pop.num_unique_inds()]
  pop_history = {"epoch":[0]*pop.elite_n, "vals": [ind.rstring for ind in pop.inds]}
  for epoch in range(epoch_n):
    # print("epoch", epoch)
    accuracies.append(pop.iterate())
    unique_inds.append(pop.num_unique_inds())
    pop_history["epoch"].extend([epoch + 1] * pop.elite_n)
    pop_history["vals"].extend([ind.rstring for ind in pop.inds])
    # yield

  name = f"{''.join(str(i) for i in trueB)}_{''.join(str(i) for i in trueS)}"
  dir = f"out/{name}(pop {pop_size}, ep {epoch_n})"
  os.makedirs(dir, exist_ok=True)
  file = open(f'{dir}/elite.csv', 'w+', newline='')
  with file:
    write = csv.writer(file)
    write.writerow([f"{e.b}_{e.s}" for e in pop.inds])
    write.writerow([e.get_rstring() for e in pop.inds])

  history = DataFrame.from_dict(pop_history)
  history.to_csv(f'{dir}/history.csv')

  fit_div = DataFrame.from_dict({"fitness": accuracies, "num_inds": unique_inds})
  fit_div.to_csv(f'{dir}/fit-div.csv')

  print("Accuracy", accuracies[-1])
  print("Best", pop.inds[0].b, pop.inds[0].s)

  # epochs = [i for i in range(0, epoch_n + 1)]
  # fig, ax1 = plt.subplots()
  #
  # color = 'tab:green'
  # ax1.set_xlabel('Epoch')
  # ax1.set_ylabel('Fitness')
  # ax1.plot(epochs, accuracies, color=color)
  # ax1.tick_params(axis='y', labelcolor=color)
  #
  # color = 'tab:red'
  # ax2 = ax1.twinx()
  # ax2.set_ylabel('Number of unique individuals')
  # ax2.set_ylim([0, pop_size])
  # ax2.plot(epochs, unique_inds, color=color)
  # ax2.tick_params(axis='y', labelcolor=color)
  #
  # fig.tight_layout()
  # plt.savefig(f'{dir}/fitness-diversity.png', bbox_inches='tight')
  # return np.mean(accuracies[-1 * (ACCURACY_EPOCH_N // 10):])


def test_density_accuracy():
  start_time = time.time()
  accuracies = [[] for _ in range(CHROMOSOME_LEN + 1)]
  for ones in range(CHROMOSOME_LEN + 1):
    goal = np.array([0] * ones + [1] * (CHROMOSOME_LEN - ones))
    for exp in range(EXPERIMENT_N):
      # print(f"Epoch {ones * EXPERIMENT_N + exp + 1}/{(CHROMOSOME_LEN + 1) * EXPERIMENT_N}")
      np.random.shuffle(goal)
      trueB = np.where(goal[:CHROMOSOME_LEN // 2] == 1)[0]
      trueS = np.where(goal[CHROMOSOME_LEN // 2:] == 1)[0]
      accuracies[ones].append(accuracy_experiment(trueB, trueS))
      yield

  print(f"Training Time: {int(time.time() - start_time)} seconds")
  print(accuracies)
  fig, ax1 = plt.subplots()
  ax1.set_xlabel('Density of Rulestring')
  ax1.set_ylabel('Accuracy Acheived')
  ax1.boxplot(accuracies)
  ax1.tick_params(axis='y')
  fig.tight_layout()
  plt.savefig(f'out/density-accuracy.png', bbox_inches='tight')


def test_initial_conds(trueB, trueS, pop_size=POPULATION_SIZE, elitism=ELITISM_RATE, mutation=MUTATION_RATE,
                       epoch_n=ACCURACY_EPOCH_N):
  pop = Population(pop_size, elitism, mutation, trueB, trueS, 'binary')
  bin_accuracies = [[] for _ in range(EXPERIMENT_N)]
  for exp in range(EXPERIMENT_N):
    bin_accuracies[exp].append(pop.evaluate(pop.loss()))
    for epoch in range(epoch_n):
      bin_accuracies[exp].append(pop.iterate())
      yield

  pop = Population(pop_size, elitism, mutation, trueB, trueS, 'decimal')
  dec_accuracies = [[] for _ in range(EXPERIMENT_N)]
  for exp in range(EXPERIMENT_N):
    dec_accuracies[exp].append(pop.evaluate(pop.loss()))
    for epoch in range(epoch_n):
      dec_accuracies[exp].append(pop.iterate())
      yield

  epochs = [i for i in range(1, epoch_n + 2)]
  fig, ax1 = plt.subplots()
  ax1.set_xlabel('Epoch')
  ax1.set_ylabel('Fitness')

  color = 'lavender'
  for i in range(len(bin_accuracies)):
    ax1.plot(epochs, bin_accuracies[i], color=color)
  ax1.plot(epochs, np.median(bin_accuracies, axis=0), color='navy')

  color = 'mistyrose'
  for i in range(len(dec_accuracies)):
    ax1.plot(epochs, dec_accuracies[i], color=color)
  ax1.plot(epochs, np.median(dec_accuracies, axis=0), color='maroon')

  ax1.tick_params(axis='y')
  fig.tight_layout()
  plt.savefig(f'out/fitness.png', bbox_inches='tight')
  return np.mean(bin_accuracies[-1 * (ACCURACY_EPOCH_N // 10):])


if __name__ == "__main__":
  # experiments = (CHROMOSOME_LEN + 1) * EXPERIMENT_N
  # with alive_bar(experiments, force_tty=True) as bar:
  #   for _ in test_density_accuracy():
  #     bar()
  for mut in (0.8,):
    for el in (0.1, 0.2):
      print("Mutation", mut)
      print("Elitism", el)
      test_single_GA({3}, {2, 3}, mutation=mut, elitism=el)
