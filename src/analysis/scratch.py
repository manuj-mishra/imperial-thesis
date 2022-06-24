import numpy as np

from lifelike.CAs import GRID_SIZE
import seaborn as sns
import matplotlib.pyplot as plt
from lifelike.constants import CHROMOSOME_LEN
from util import binary

if __name__ == "__main__":
  # rstrings = np.random.randint(0, 2 ** CHROMOSOME_LEN - 1, size=100)
  # np.save('lifelike/goals.npy', rstrings)

  # with open('equal_density_goals/goals.npy', 'rb') as goalfile:
  #   goals = np.load(goalfile)
  #
  # print(goals)
  # densgoal = [bin(x).count("1") for x in goals]
  # sns.histplot(x=densgoal, kde=True, stat='frequency')
  # plt.show()

  # manyGoals = []
  # while len(manyGoals) < 1000:
  #   d = np.random.uniform(0, 1)
  #   X = np.random.random((13, 13)) < d
  #   manyGoals.append(X)
  #
  #
  # goals = np.unique(manyGoals, axis=0)[:100]
  # np.save('lifelike/ics.npy', goals)

  rstring = 256

  print(binary.ones(rstring >> (CHROMOSOME_LEN // 2)))
  print(binary.ones(rstring))
