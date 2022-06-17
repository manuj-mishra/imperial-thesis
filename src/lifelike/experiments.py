import csv
from random import random

import numpy as np
import pandas as pd

from lifelike.CAs import GRID_SIZE
from lifelike.Population import Population
from lifelike.constants import CHROMOSOME_LEN
from util import binary
#HYPERPARAMETER TUNING
if __name__ == "__main__":
  ics = [np.random.random((GRID_SIZE, GRID_SIZE)) > np.random.random() for i in range(20)]
  goals = [np.random.randint(1, 2 ** CHROMOSOME_LEN - 1) for _ in range(100)]


  # pop_size = 20
  # elitism = 0.2
  # mutation = 1 / CHROMOSOME_LEN
  # epoch_n = 30
  # hyperparams = {
  #   "max_step": 5,
  #   "eval_step": 10
  # }
  # exp_name = f"exp_maxstep{max_step}_evalstep{eval_step}"
  # print(f"Running {exp_name}")
  # rules = []
  # conv_epochs = []
  # num_visited = []
  # best_rules = []
  # pcount = 0
  # for goalarr in goals:
  #   print(pcount)
  #   rules.append(goalarr)
  #   trueB = binary.ones(goalarr >> (CHROMOSOME_LEN // 2))
  #   trueS = binary.ones(goalarr)
  #   pop = Population(pop_size, elitism, mutation, trueB, trueS, ics, 'binary', hyperparams)
  #   counter = 0
  #   for _ in range(epoch_n):
  #     if pop.goal_found():
  #       break
  #     pop.iterate()
  #     counter += 1
  #   conv_epochs.append(counter)
  #   num_visited.append(len(pop.visited))
  #   best_rules.append(pop.inds[0].rstring)
  #   pcount += 1
  #
  # df = pd.DataFrame({"rstring": rules,
  #                    "convtime": conv_epochs,
  #                    "visited": num_visited,
  #                    "bestrule": best_rules})
  # df.to_csv(f"./{exp_name}.csv")
