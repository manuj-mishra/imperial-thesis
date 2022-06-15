import csv
import os
import numpy as np
from matplotlib import pyplot as plt

from gray_scott.CAs import CA
from gray_scott.Population import Population
from util.media import create_conv_gif

root = '.'

# Repetition variables
ACCURACY_EPOCH_N = 30  # Epochs for accuracy experiment

# Training variables
N_PARENTS = 5
N_CHILDREN = N_PARENTS * 4

def test_single_EA(true_f, true_k, rname, algorithm, recombination, selection, initialisation, seed,
                   n_parents=N_PARENTS, n_children=N_CHILDREN, epoch_n=ACCURACY_EPOCH_N):
    pop = Population(n_parents, n_children, true_f, true_k, algorithm, recombination, selection, initialisation)
    # losses = [pop.evaluate(pop.loss())]
    top_f = [pop.inds[0].state[0]]
    top_k = [pop.inds[0].state[1]]
    top_df = [pop.inds[0].control[0]]
    top_dk = [pop.inds[0].control[1]]

    fig, ax = plt.subplots()

    for epoch in range(1, epoch_n + 1):
        print(epoch)
        loss = pop.iterate()
        # losses.append(loss)
        top_f.append(pop.inds[0].state[0])
        top_k.append(pop.inds[0].state[1])
        top_df.append(pop.inds[0].control[0])
        top_dk.append(pop.inds[0].control[1])

        ax.scatter([i.state[1] for i in pop.inds], [i.state[0] for i in pop.inds], c='b')
        ax.scatter(true_k, true_f, c='r')
        ax.set_xlabel("Kill")
        ax.set_ylabel("Feed")
        ax.set_xlim([-0.01, 0.30])
        ax.set_ylim([-0.01, 0.08])
        plt.savefig(f'{root}/temp/conv_frames/{epoch}.png', bbox_inches='tight')
        plt.cla()

    plt.close(fig)

    dir = f"{root}/out/{rname}"
    os.makedirs(dir, exist_ok=True)

    lines = [f'Experiment {rname}',
             f'Algorithm {algorithm}',
             f'Recombination {recombination}',
             f'Selection {selection}',
             f'Initialisation{initialisation}',
             f'Seed {seed}'
             ]
    with open(f'{dir}/config.txt', 'w') as f:
        f.write('\n'.join(lines))

    file = open(f'{dir}/elite.csv', 'w+', newline='')
    with file:
        write = csv.writer(file)
        for e in pop.inds:
            write.writerow(
                [f"f:{e.state[0]:.3f} k:{e.state[1]:.3f} df:{e.control[0]:.3f} dk:{e.control[1]:.3f}" for e in
                 pop.inds])

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

    # fig, ax = plt.subplots()
    # ax.plot(epochs, losses, color='tab:blue')
    # ax.set_yscale('log')
    # ax.set_ylabel("Loss")
    # plt.savefig(f'{dir}/loss.png', bbox_inches='tight')
    # plt.close(fig)

    create_conv_gif(rname=rname)

    if seed == "PATCH":
        new_CA = CA.patch
    elif seed == "SPLATTER":
        new_CA = CA.splatter
    else:
        raise Exception("Invalid argument for CA seed type")

    goal = new_CA(f=true_f, k=true_k)
    goal.run(fname="goal", rname=rname, media=True)

    found = new_CA(f=top_f[-1], k=top_k[-1])
    found.run(fname="pred", rname=rname, media=True)

    # return np.mean(losses[-1 * (ACCURACY_EPOCH_N // 10):])


if __name__ == "__main__":
    # np.set_printoptions(precision=3, suppress=True)
    test_single_EA(true_f=0.03, true_k=0.06, rname="GA_30EP_25POP", algorithm="GA", recombination="PLUS",
                   selection="LINEAR", initialisation="RANDOM", seed="SPLATTER")
