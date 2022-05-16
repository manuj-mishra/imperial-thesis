import csv
import os
import numpy as np
from matplotlib import pyplot as plt

from gray_scott_simple.CAs import CA
from gray_scott_simple.Population import Population
from util.media import create_conv_gif

# Repetition variables

ACCURACY_EPOCH_N = 15  # Epochs for accuracy experiment
CONV_MAX_EPOCH_N = 20  # Epochs for convergence experiment

# Training variables
N_PARENTS = 4
N_CHILDREN = 16


def accuracy_experiment(true_f, true_k, n_parents=N_PARENTS, n_children=N_CHILDREN, epoch_n=ACCURACY_EPOCH_N):
  pop = Population(n_parents, n_children, true_f, true_k)
  for i in range(epoch_n - 1):
    print(i, pop.iterate())
  print(epoch_n, pop.iterate())
  return pop.iterate()


def test_single_ES(true_f, true_k, n_parents=N_PARENTS, n_children=N_CHILDREN, epoch_n=ACCURACY_EPOCH_N):
  pop = Population(n_parents, n_children, true_f, true_k)
  losses = [pop.evaluate(pop.loss())]
  top_f = [pop.inds[0].state[0]]
  top_k = [pop.inds[0].state[1]]
  top_df = [pop.inds[0].control[0]]
  top_dk = [pop.inds[0].control[1]]

  fig, ax = plt.subplots()
  ax.scatter([i.state[0] for i in pop.inds], [i.state[1] for i in pop.inds])
  ax.set_xlabel("Feed")
  ax.set_ylabel("Kill")
  ax.set_xlim([-0.01, 0.1])
  ax.set_ylim([-0.01, 0.1])
  plt.savefig(f'gray_scott_simple/temp/conv_frames/0.png', bbox_inches='tight')
  plt.cla()

  for epoch in range(1, epoch_n + 1):
    loss = pop.iterate()
    losses.append(loss)
    top_f.append(pop.inds[0].state[0])
    top_k.append(pop.inds[0].state[1])
    top_df.append(pop.inds[0].control[0])
    top_dk.append(pop.inds[0].control[1])

    ax.scatter([i.state[0] for i in pop.inds], [i.state[1] for i in pop.inds])
    ax.set_xlabel("Feed")
    ax.set_ylabel("Kill")
    ax.set_xlim([-0.01, 0.1])
    ax.set_ylim([-0.01, 0.1])
    plt.savefig(f'gray_scott_simple/temp/conv_frames/{epoch}.png', bbox_inches='tight')
    plt.cla()

  plt.close(fig)

  rname = f"{true_f:.3f}_{true_k:.3f}_(pop_{n_parents + n_children},_ep_{epoch_n})"
  dir = f"gray_scott_simple/out/{rname}"
  os.makedirs(dir, exist_ok=True)
  file = open(f'{dir}/elite.csv', 'w+', newline='')
  with file:
    write = csv.writer(file)
    for e in pop.inds:
      write.writerow(
        [f"f:{e.state[0]:.3f} k:{e.state[1]:.3f} df:{e.control[0]:.3f} dk:{e.control[1]:.3f}" for e in pop.inds])

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
  plt.close(fig)

  fig, ax1 = plt.subplots()
  color = 'tab:green'
  ax1.set_xlabel('Epoch')
  ax1.set_ylabel('delta_F')
  ax1.plot(epochs, top_df, color=color)
  ax1.tick_params(axis='y', labelcolor=color)
  color = 'tab:red'
  ax2 = ax1.twinx()
  ax2.set_ylabel('delta_K')
  ax2.plot(epochs, top_dk, color=color)
  ax2.tick_params(axis='y', labelcolor=color)
  fig.tight_layout()
  plt.savefig(f'{dir}/derivs.png', bbox_inches='tight')
  plt.close(fig)

  fig, ax = plt.subplots()
  ax.plot(epochs, losses, color='tab:blue')
  ax.set_yscale('log')
  ax.set_ylabel("Loss")
  plt.savefig(f'{dir}/loss.png', bbox_inches='tight')
  plt.close(fig)

  create_conv_gif(rname=rname)

  goal = CA.patch(f=true_f, k=true_k)
  goal.run(fname="goal", rname=rname, media=True)

  found = CA.patch(f=top_f[-1], k=top_k[-1])
  found.run(fname="pred", rname=rname, media=True)


  return np.mean(losses[-1 * (ACCURACY_EPOCH_N // 10):])


if __name__ == "__main__":
  # rname = f"{0.038:.3f}_{0.099:.3f} (pop {N_PARENTS + N_CHILDREN}, ep {ACCURACY_EPOCH_N})"
  # create_conv_gif(rname=rname)
  # accuracy_experiment(true_f=0.038, =0.099)
  np.set_printoptions(precision=3, suppress=True)
  test_single_ES(true_f=0.038, true_k=0.099)

