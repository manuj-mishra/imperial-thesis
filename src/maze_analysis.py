import os
from functools import reduce
from random import random
import re
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats

if __name__ == "__main__":

  df = pd.read_csv('./maze_hyperparam.csv')
  df = df.loc[(df.d + df.p).sort_values(ascending=False).index]
  print(df)
  g = sns.scatterplot(y=df.d/df.t, x = df.p/df.t)
  g.set(xlabel='Path length / Time taken', ylabel='# of dead ends / Time taken')
  plt.show()