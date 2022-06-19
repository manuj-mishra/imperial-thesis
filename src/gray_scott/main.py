import csv
import os
import numpy as np
from matplotlib import pyplot as plt
from pandas import DataFrame

from gray_scott.CAs import CA
from gray_scott.Population import Population
from util.media import create_conv_gif

root = '.'

# Repetition variables
ACCURACY_EPOCH_N = 30  # Epochs for accuracy experiment

# Training variables
N_PARENTS = 5
N_CHILDREN = N_PARENTS * 4

def test_single_EA(true_f, true_k, algorithm, recombination, selection, initialisation, seed,
                   n_parents=N_PARENTS, n_children=N_CHILDREN, epoch_n=ACCURACY_EPOCH_N):
    pop = Population(n_parents, n_children, true_f, true_k, algorithm, recombination, selection, initialisation, seed)
    # losses = [pop.evaluate(pop.loss())]
    top_f = [pop.inds[0].state[0]]
    top_k = [pop.inds[0].state[1]]
    # top_df = [pop.inds[0].control[0]]
    # top_dk = [pop.inds[0].control[1]]

    for epoch in range(1, epoch_n + 1):
        print(epoch)
        loss = pop.iterate()
        # losses.append(loss)
        top_f.append(pop.inds[0].state[0])
        top_k.append(pop.inds[0].state[1])
        # top_df.append(pop.inds[0].control[0])
        # top_dk.append(pop.inds[0].control[1])

    # dir = f"{root}/out/{rname}"
    # os.makedirs(dir, exist_ok=True)
    #
    # lines = [f'Experiment {rname}',
    #          f'Algorithm {algorithm}',
    #          f'Recombination {recombination}',
    #          f'Selection {selection}',
    #          f'Initialisation{initialisation}',
    #          f'Seed {seed}'
    #          ]
    # with open(f'{dir}/config.txt', 'w') as f:
    #     f.write('\n'.join(lines))
    #
    # file = open(f'{dir}/elite.csv', 'w+', newline='')
    # with file:
    #     write = csv.writer(file)
    #     write.writerow(
    #             [f"f:{e.state[0]:.3f} k:{e.state[1]:.3f}" for e in
    #              pop.inds])
    #
    # epochs = [i for i in range(0, epoch_n + 1)]
    # DataFrame.from_dict({"t": epochs, "f": top_f, "k": top_k}).to_csv("res.csv")
    #
    # if seed == "PATCH":
    #     new_CA = CA.patch
    # elif seed == "SPLATTER":
    #     new_CA = CA.splatter
    # else:
    #     raise Exception("Invalid argument for CA seed type")
    #
    # goal = new_CA(f=true_f, k=true_k)
    # goal.run(fname="goal", rname=rname, media=True)
    #
    # found = new_CA(f=top_f[-1], k=top_k[-1])
    # found.run(fname="pred", rname=rname, media=True)

    # return np.mean(losses[-1 * (ACCURACY_EPOCH_N // 10):])

    return top_f, top_k

if __name__ == "__main__":
    res = {"fs":[], "ks":[], "recomb": [], "select": [], "initi": [], "seed": []}
    for recomb in ("PLUS", "COMMA"):
        for select in ("LINEAR", "ROULETTE"):
            for initi in ("RANDOM", "THRESHOLD"):
                for seed in ("SPLATTER", "PATCH"):
                  rname = f"GA_{recomb}_{select}_{initi}_{seed}"
                  f,k = test_single_EA(true_f=0.03, true_k=0.06, algorithm="GA", recombination=recomb,
                                 selection=select, initialisation=initi, seed=seed)
                  res["fs"].append(f)
                  res["ks"].append(k)
                  res["recomb"].append(recomb)
                  res["select"].append(select)
                  res["initi"].append(initi)
                  res["seed"].append(seed)
                  DataFrame.from_dict(res).to_csv("chonkyboi.csv")
