import os
from functools import reduce
from random import random
import re
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats

def taxonomy_graphs(df):
  data = df.conv_perc
  graph(data, title="", xlab="Convergence Proportion")

def graph(data, title, xlab, ylab="Frequency Density"):
  sns.set(style="darkgrid")
  f, (ax_box, ax_hist) = plt.subplots(2, sharex='all', gridspec_kw={"height_ratios": (.15, .85)})
  b = sns.boxplot(x=data, ax=ax_box)
  h = sns.histplot(x=data, ax=ax_hist, stat='frequency',binwidth=0.01)
  ax_box.set(xlabel='', title=title)
  ax_hist.set(xlabel=xlab, ylabel=ylab)
  plt.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
  plt.ylim((0, 8e5))
  plt.show()

def eppstein(df):

  ones = sum(df.conv_perc == 1)
  df = df[(df.rstring & 114688) == 0]
  print(len(df) / ones)
  ones = sum(df.conv_perc == 1)
  print(ones / len(df))
  # df = df[df.conv_mean != 1]
  h = sns.histplot(x=df.conv_mean, kde=True, stat='frequency')
  h.set(xlabel="Time steps to convergence",  ylabel="Frequency Density", title="Convergence of rules not including B1, B2, or B3")
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

def runs_vs_convperc(root = "lifelike/single-hyper"):
  d = {'maxstep': [], 'evalstep': [], 'convergence':[], 'visited':[], 'convtime':[]}
  for file in os.listdir(root):
    maxstep = float(re.search('max(.*)_eval', file).group(1))
    evalstep = float(re.search('eval(.*).csv', file).group(1))
    d['maxstep'].append(maxstep)
    d['evalstep'].append(evalstep)
    locdf = pd.read_csv(f"./{root}/{file}")
    locdf = locdf[locdf.rstring == locdf.bestrule]
    d['convergence'].append(len(locdf.index)/100)
    d['visited'].append(sum(locdf.visited)/(len(locdf.index)))
    d['convtime'].append(sum(locdf.convtime) / len(locdf.index))

  for i in range(len(d['maxstep'])):
    print(d['maxstep'][i], d['evalstep'][i], d['convergence'][i], d['convtime'][i], d['visited'][i])
  # dfs = []
  # for file in os.listdir(root):
  #     df = pd.read_csv(f"./{root}/{file}")
  #     df["type"] = file[:-4]
  #     dfs.append(df)
  # bigboi = pd.concat(dfs, ignore_index=True)
  # g = sns.kdeplot(data=bigboi, x='convtime', hue='type', label=False)
  # plt.legend(loc='upper right', title='Type', labels=['Multi-Resolution Loss', 'Single Resolution Loss'])
  # g.set(xlabel='Convergence Time')
  # plt.show()

  # sns.kdeplot(single.convtime)
  # sns.kdeplot(multi.convtime)
  # plt.show()

  # df = pd.DataFrame(data=d)
  # print(d)
  # g = sns.lineplot(data=df, x="maxstep", y="convergence", hue="evalstep", palette="Set2")
  # g.set(xscale='log', xlabel='Maximum step size', ylabel='Proportion of goals precisely learnt in under 30 epochs', title='Tuning maximum step size')
  # plt.show()
  # plt.cla()
  #
  #
  # h = sns.histplot(x=df.convtime, kde=True, stat='frequency')
  # h.set(xscale='log', xlabel='Convergence Time', ylabel='Frequency Density', title='Distribution of Convergence')
  # plt.show()
  # h = sns.lineplot(data=df, x="maxstep", y="visited", hue="evalstep", palette="Set2")
  # h.set(xscale='log', xlabel='Maximum step size', ylabel='Proportion of rule space traversed', title='')
  # plt.show()


def tuning_max_step_size():
  d = {'stepsize': [1, 10, 100, 1, 10, 100], 'population': [10, 10, 10, 100, 100, 100], 'convergence':[.19, .16, .17, .2, .32, .25]}
  df = pd.DataFrame(data=d)
  g = sns.scatterplot(data=df, x="stepsize", y="convergence", hue="population", palette="Set2", s=100)
  g.set(xscale='log', xlabel='Maximum step size', ylabel='Proportion of goals precisely learnt in under 30 epochs', title='Tuning maximum step size')
  # plt.ylim(.15, .35)
  plt.show()

def low_high_sparsity():
  df = pd.read_csv('./taxonomy.csv')
  df = df[(df.period_mean <= 2) & (df.period_mean >= 1)]
  density = df.rstring.apply(lambda x: bin(x).count("1"))
  g = sns.kdeplot(x=df.period_mean, y=density)
  # g.set(xscale='log', xlabel='Maximum step size', ylabel='Proportion of goals precisely learnt in under 30 epochs', title='Tuning maximum step size')
  # plt.xlim(1, 2)
  plt.show()



if __name__ == "__main__":
  # df = pd.read_csv('./lifelike/random_negentropy.csv')
  # sns.histplot(df.negentropy)
  # plt.show()
  # negdict = dict(zip(df.rstring, -1 * df.negentropy))
  # # print(negdict[8])
  # print(negdict[int('000100000001100000', 2)])
  # print(pd.isna(negdict[0]))
  # print(df.negentropy.quantile(.95))
  # print(df.conv_perc)


  # df = pd.read_csv('./gray_scott/res.csv')
  # sns.color_palette("flare", as_cmap=True)
  # df = df[df.t > 10]
  # g = sns.lineplot(x=df.k, y=df.f, sort=False)
  # plt.show()

  # runs_vs_convperc()


  with open('lifelike/ics.npy', 'rb') as file:
    ics = np.load(file)

  hyp = pd.read_csv('./lifelike/single-hyper/single_max5_eval5.csv')
  tax = pd.read_csv('./taxonomy.csv')
  tax = tax[tax.conv_perc != 0]
  tax['type'] = pd.cut(tax.conv_perc, bins= [0.0, 0.99, 1], labels=['Some', 'All'])
  tax = tax.drop(columns=['conv_perc', 'conv_std', 'period_std', 'entropy', 'rstring'])
  sns.pairplot(tax, plot_kws={'fill':True}, kind='kde', diag_kind='kde', hue='type')
  plt.show()
  # hyp = hyp.set_index('rstring')
  # hyp['conv_perc'] = tax.conv_perc
  # hyp['conv_mean'] = tax.conv_mean
  # hyp['period_mean'] = tax.period_mean
  # hyp['density'] = tax.density
  # hyp['volatility'] = tax.volatility
  # print(df.conv_perc.value_counts())
  # df = odf[odf.conv_perc != 0]
  # df = df[df.conv_perc != 1]
  # print(len(df.index))
  # print(len(df.index))
  # graph(df.entropy, "", "")
  # taxonomy_graphs(df)
  # print(hyp)
  # for thing in ('conv_perc', 'conv_mean', 'period_mean', 'density', 'volatility'):
  #   sns.scatterplot(data=hyp, x = 'visited', y =thing)
  #   plt.show()
  #   plt.cla()