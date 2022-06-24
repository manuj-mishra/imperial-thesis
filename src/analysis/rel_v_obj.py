import os
from functools import reduce
from random import random
import re
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pandas import DataFrame
from scipy import stats

if __name__ == "__main__":
  df = pd.read_csv('../maze/obj.csv')
  all_dict = {"Epoch":[], "Obj":[]}
  for epoch in range(31):
    all_dict['Epoch'].extend([epoch]*100)
    all_dict['Obj'].extend(df[str(epoch)])
  df2 = DataFrame.from_dict(all_dict)
  g = sns.scatterplot(data=df2, x='Epoch', y='Obj', marker='x')
  # g.set(xscale='log', xlabel='Maximum step size', ylabel='Proportion of goals precisely learnt in under 30 epochs', title='Tuning maximum step size')
  # plt.xlim(1, 2)
  plt.show()
