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
  # trunc = pd.read_csv('./rel_truncation.csv')
  # obj = pd.read_csv('./roulette_ranks.csv')
  # df = pd.concat([obj, trunc], ignore_index=True)
  # g = sns.kdeplot(x=.5 * (df.p + df.d), hue = df.type, legend=False, fill=True)
  # plt.legend(loc='upper left', title='Type', labels=['Relative Truncation', 'Relative Roulette'])
  # g.set(xlabel='Fitness')
  # plt.show()

  # df = pd.read_csv('./maze_hyperparam.csv')
  # df = df.loc[(df.d + df.p).sort_values(ascending=False).index]
  # print(df)
  # g = sns.scatterplot(y=df.d, x=df.p)
  # g.set(xlabel='Path length', ylabel='# of dead ends')
  # plt.show()

  # both = pd.read_csv('./maze/maze-both.csv')
  # both = both.loc[:, ~both.columns.str.match('Unnamed')]
  # mut = pd.read_csv('./maze/maze-no-mutation.csv')
  # mut = mut.loc[:, ~mut.columns.str.match('Unnamed')]
  # cross = pd.read_csv('./maze/maze-no-crossover.csv')
  # nothing = pd.read_csv('./maze/maze-nothing.csv')
  # both.category = both.category.astype(str) + " mutation and crossover"
  # # print(both)
  # df = pd.concat([nothing, cross, mut, both], ignore_index=True)
  # print(min(df.d), min(df.p))
  # plt.ylim(0, 110)
  # plt.xlim(60, 85)
  # g = sns.scatterplot(data=cross, y="d", x = "p", color='r')
  # g.set(xlabel='Path length', ylabel='# of dead ends')
  # plt.show()
  # plt.cla()
  # plt.ylim(0, 110)
  # plt.xlim(60, 85)
  # g = sns.scatterplot(data=nothing, y="d", x = "p", color='b')
  # g.set(xlabel='Path length', ylabel='# of dead ends')
  # plt.show()
  # plt.cla()
  # plt.ylim(0, 110)
  # plt.xlim(60, 85)
  # g = sns.scatterplot(data=mut, y="d", x = "p", color='c')
  # g.set(xlabel='Path length', ylabel='# of dead ends')
  # plt.show()
  # plt.cla()
  # plt.ylim(0, 110)
  # plt.xlim(60, 85)
  # g = sns.scatterplot(data=both, y="d", x = "p", color='g')
  # g.set(xlabel='Path length', ylabel='# of dead ends')
  # plt.show()
  # plt.cla()

  # plt.legend(loc='lower right')
  # sns.scatterplot(data=df, y="d", x = "p", hue="category")
  # plt.show()
