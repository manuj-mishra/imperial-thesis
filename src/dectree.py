import os
from functools import reduce
from random import random

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats

def taxonomy_graphs(df):
  N = 262144
  df.b1 = (df.rstring & (1 << 17)) != 0
  print(df.b1)
  df = df[df.b1 != 1]
  data = df.conv_perc
  print(data.describe())
  graph(data, title="", xlab="")

def taxonomy_graphs(df):
  N = 262144
  df.b1 = (df.rstring & (1 << 17)) != 0
  print(df.b1)
  df = df[df.b1 != 1]
  data = df.conv_perc
  print(data.describe())
  graph(data, title="", xlab="")

def graph(data, title, xlab, ylab="Frequency Density"):
  sns.set(style="darkgrid")
  f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
  b = sns.boxplot(data, ax=ax_box)
  h = sns.histplot(x=data, ax=ax_hist, kde=True, stat='frequency')
  ax_box.set(xlabel='', title=title)
  ax_hist.set(xlabel=xlab, ylabel=ylab)
  plt.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
  plt.show()

def eppstein(df):

  ones = sum(df.conv_perc == 1)
  df = df[(df.rstring & 114688) == 0]
  print(len(df) / ones)
  ones = sum(df.conv_perc == 1)
  print(ones / len(df))
  # df = df[df.conv_mean != 1]
  h = sns.histplot(x=df.conv_mean, kde=True, stat='frequency')
  h.set(xlabel="Time steps to convergence", ylabel="Frequency Density", title="Convergence of rules not including B1, B2, or B3")
  sns.set(style="darkgrid")

  plt.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
  plt.show()
# df.density = df.rstring.apply(lambda x: bin(x).count("1"))
#
# # # Percent plot
# # sns.displot(df.density, discrete=True, stat='percent')
#
# g = sns.jointplot(x = df.density, y=df.conv_perc, marker="+")
# g.plot_joint(sns.kdeplot, fill=True)
# g.plot_marginals(sns.displot, color="r")
#
# plt.show()
#
#
# dist = getattr(stats, 'norm')
# parameters = dist.fit(df.density)
# print(parameters)
#


if __name__ == "__main__":
  # df = pd.read_csv('./taxonomy.csv')
  # taxonomy_graphs(df)

  # for file in os.listdir("./"):
  #   if file.startswith('step') and file.endswith(".csv"):
  #     df = pd.read_csv(f"./{file}")
  #     print(file)
  #     df = df.loc[:, df.iloc[0] != 30]
  #     print(len(df.iloc[0]) / 100)
      # print(max(df.iloc[0]))
      # print((100*sum(df.iloc[1]))/(converged*2**18))

  stepsize = [1, 10, 100]
  d = {'stepsize': [1, 10, 100, 1, 10, 100], 'population': [10, 10, 10, 100, 100, 100], 'convergence':[.19, .16, .17, .2, .32, .25]}
  df = pd.DataFrame(data=d)
  g = sns.scatterplot(data=df, x="stepsize", y="convergence", hue="population", palette="Set2", s=100)
  g.set(xscale='log', xlabel='Maximum step size', ylabel='Proportion of goals precisely learnt in under 30 epochs', title='Tuning maximum step size')
  # plt.ylim(.15, .35)
  plt.show()