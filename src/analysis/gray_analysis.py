import os
from ast import literal_eval
from functools import reduce
from random import random
import re

import np as np
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats

if __name__ == "__main__":

  df = pd.read_csv('chonkyboi.csv')
  print(df.columns)
  fs = []
  for i in df.fs:
    fs.extend(literal_eval(i))
  ks = []
  for i in df.ks:
    ks.extend(literal_eval(i))
  sns.scatterplot(x=ks, y=fs, marker='x', hue=np.repeat(df.seed, 31))
  xs = np.linspace(0, 0.0625)
  ys1 = 0.125 * ((-8 * xs) + np.sqrt(1 - 16 * xs) + 1)
  ys2 = 0.125 * ((-8 * xs) - np.sqrt(1 - 16 * xs) + 1)
  sns.lineplot(x=xs, y=ys1, color='k')
  g = sns.lineplot(x=xs, y=ys2, color='k')
  g.set(xlabel='k', ylabel='f')

  sns.scatterplot(x=[0.062], y=[0.055], color='y', marker='o', s=100)

  plt.show()