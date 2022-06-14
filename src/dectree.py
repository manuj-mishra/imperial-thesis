import os
from functools import reduce
from random import random

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree
import graphviz


def graph(data, title, xlab, ylab="Frequency Density"):
  sns.set(style="darkgrid")
  f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
  b = sns.boxplot(data, ax=ax_box)
  h = sns.histplot(x=data, ax=ax_hist, kde=True, stat='frequency')
  ax_box.set(xlabel='', title=title)
  ax_hist.set(xlabel=xlab, ylabel=ylab)
  plt.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
  plt.show()

if __name__ == "__main__":
  df = pd.read_csv("./stepsize10_pop100.csv").transpose()
  df = df.reset_index(level=0)
  df = df.rename(columns={"index":"rstring"})
  df = df.astype({'rstring': 'float64'})
  # df = df.drop(df[df.rstring % 1 != 0].index)
  df = df.astype('int64')
  for i in range(9):
    df[f'b{i}'] = df.rstring.apply(lambda rstring: ((rstring & (1 << i+9)) != 0))
  for i in range(9):
    df[f's{i}'] = df.rstring.apply(lambda rstring: ((rstring & (1 << i)) != 0))

  y = df[0] == 30
  print(y)
  X = df.drop(columns=['rstring',0, 1])
  clf = DecisionTreeClassifier(random_state=1234, max_depth=3)
  model = clf.fit(X, y)
  # DOT data
  dot_data = tree.export_graphviz(clf, out_file=None,
                                  feature_names=X.columns,
                                  class_names=['Success', 'Failure'],
                                  filled=True)
  graph = graphviz.Source(dot_data, format="png")
  graph.render("decision_tree_graphivz")



  