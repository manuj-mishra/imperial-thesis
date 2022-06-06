import csv
from random import random

import numpy as np

from lifelike.Population import Population
from lifelike.constants import CHROMOSOME_LEN

if __name__ == "__main__":
  pop_size = 100
  max_step = 10
  elitism = 0.2
  mutation = 1 / CHROMOSOME_LEN
  epoch_n = 30
  num_rules = 100
  hyperparams = {
    "max_step": max_step,
    "eval_step": 10
  }
  exp_name = f"epoch{epoch_n}_pop{pop_size}"
  print(f"Running {exp_name}")
  rules = []
  conv_epochs = []
  num_visited = []
  for rule_num in range(num_rules):
    print(f"Rule {rule_num}/{num_rules}")
    binarr = np.random.binomial(1, np.random.uniform(low = 1/18, high=17/18), size=CHROMOSOME_LEN)
    binarrstr = ''.join(binarr.astype(str))
    rules.append(int(binarrstr, 2))
    trueB = np.where(binarr[:CHROMOSOME_LEN // 2] == 1)[0]
    trueS = np.where(binarr[CHROMOSOME_LEN // 2:] == 1)[0]
    with open('lifelike/ics.npy', 'rb') as icfile:
      ics = np.load(icfile)
    pop = Population(pop_size, elitism, mutation, trueB, trueS, ics, 'binary', hyperparams)
    counter = 0
    for _ in range(epoch_n):
      if pop.goal_found():
        break
      pop.iterate()
      counter += 1
    conv_epochs.append(counter)
    num_visited.append(len(pop.visited))
  with open(f"./{exp_name}.csv", 'w', newline='') as data:
    wr = csv.writer(data, quoting=csv.QUOTE_ALL)
    wr.writerow(rules)
    wr.writerow(conv_epochs)
    wr.writerow(num_visited)
