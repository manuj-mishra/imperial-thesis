import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

if __name__ == "__main__":
  df = pd.read_csv('report/life-nothing.csv')
  g = sns.scatterplot(data=df, x="epoch", y="vals", s = 20, marker='x')
  g.set(xlabel='Epoch', ylabel='Integer value of chromosome')
  plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
  rtrue = int('001000000001100000', 2)
  plt.axhline(y=rtrue, color='r')
  ax = plt.gca()
  ax.set_ylim(ymin = -10, ymax=5000)
  plt.show()
  # plt.cla()
  # df = df[df['epoch'] == 30]
  # h = sns.ecdfplot(data=df, x='vals')
  # h.set(xlabel='Integer value of chromosome')
  # # ax.set_xlim(xmin = -10, xmax=1000)
  # # plt.axvline(x=32864, color='r')
  # plt.show()
  print(df.vals.value_counts())