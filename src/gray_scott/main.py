import csv
import os
import time
import numpy as np
from matplotlib import pyplot as plt

from gray_scott.CAs import CA
from gray_scott.Population import Population
from alive_progress import alive_bar

# Repetition variables
EXPERIMENT_N = 10  # Experiments per density evaluated
ACCURACY_EPOCH_N = 30  # Epochs for accuracy experiment
CONV_MAX_EPOCH_N = 20  # Epochs for convergence experiment

# Training variables
POPULATION_SIZE = 30
ELITISM_RATE = 0.5
ALPHA = 0.1


def accuracy_experiment(true_f, true_k, pop_size=POPULATION_SIZE, elitism=ELITISM_RATE, alpha=ALPHA,
                        epoch_n=ACCURACY_EPOCH_N):
  pop = Population(pop_size, elitism, alpha, true_f, true_k)
  for i in range(epoch_n - 1):
    print(i, pop.iterate())
  print(epoch_n, pop.iterate())
  return pop.iterate()


def test_single_GA(true_f, true_k, pop_size=POPULATION_SIZE, elitism=ELITISM_RATE, alpha=ALPHA,
                   epoch_n=ACCURACY_EPOCH_N):
  pop = Population(pop_size, elitism, alpha, true_f, true_k)
  losses = [pop.evaluate(pop.loss())]
  top_f = [pop.inds[0].f]
  top_k = [pop.inds[0].k]
  top_df = [pop.inds[0].df]
  top_dk = [pop.inds[0].dk]
  for epoch in range(epoch_n):
    loss = pop.iterate()
    print(loss, pop.inds[0].f, pop.inds[0].k)
    losses.append(loss)
    top_f.append(pop.inds[0].f)
    top_k.append(pop.inds[0].k)
    top_df.append(pop.inds[0].df)
    top_dk.append(pop.inds[0].dk)
    yield

  rname = f"{true_f:.3f}_{true_k:.3f} (pop {pop_size}, ep {epoch_n})"
  dir = f"out/{rname}"
  os.makedirs(dir, exist_ok=True)
  file = open(f'{dir}/elite.csv', 'w+', newline='')
  with file:
    write = csv.writer(file)
    for e in pop.inds:
      write.writerow([f"f:{e.f:.3f} k:{e.k:.3f} df:{e.df:.3f} dk:{e.dk:.3f}" for e in pop.inds])

  epochs = [i for i in range(0, epoch_n + 1)]
  fig, ax1 = plt.subplots()

  color = 'tab:green'
  ax1.set_xlabel('Epoch')
  ax1.set_ylabel('F')
  ax1.plot(epochs, top_f, color=color)
  ax1.tick_params(axis='y', labelcolor=color)

  color = 'tab:red'
  ax2 = ax1.twinx()
  ax2.set_ylabel('K')
  ax2.plot(epochs, top_k, color=color)
  ax2.tick_params(axis='y', labelcolor=color)

  fig.tight_layout()
  plt.savefig(f'{dir}/params.png', bbox_inches='tight')


  color = 'tab:green'
  ax1.set_xlabel('Epoch')
  ax1.set_ylabel('dF')
  ax1.plot(epochs, top_df, color=color)
  ax1.tick_params(axis='y', labelcolor=color)

  color = 'tab:red'
  ax2 = ax1.twinx()
  ax2.set_ylabel('dK')
  ax2.plot(epochs, top_dk, color=color)
  ax2.tick_params(axis='y', labelcolor=color)

  fig.tight_layout()
  plt.savefig(f'{dir}/derivs.png', bbox_inches='tight')

  fig, ax = plt.subplots()
  ax.plot(epochs, losses, color='tab:blue')
  ax.set_ylabel("Loss")
  plt.savefig(f'{dir}/loss.png', bbox_inches='tight')

  goal = CA.patch(f=true_f, k=true_k)
  goal.run(fname="goal", rname=rname, media=True)

  found = CA.patch(f=top_f[-1], k=top_k[-1])
  found.run(fname="pred", rname=rname, media=True)

  return np.mean(losses[-1 * (ACCURACY_EPOCH_N // 10):])

if __name__ == "__main__":
  with alive_bar(ACCURACY_EPOCH_N, force_tty=True) as bar:
    for _ in test_single_GA(true_f=0.038, true_k=0.099):
      bar()
