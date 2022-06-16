import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

if __name__ == "__main__":
  rtrue = int('00100000001100000', 2)
  df = pd.read_csv('./out/3_23(pop 20, ep 30)/history.csv')
  y = abs(df.vals - rtrue)
  g = sns.scatterplot(data=df, x="epoch", y=y, s = 20)
  g.set(xlabel='Epoch', ylabel='Difference in integer value of chromosome from true value')
  plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
  # plt.axhline(y=rtrue, color='r')
  ax = plt.gca()
  ax.set_ylim(ymin = -1000, ymax=2**18)
  plt.show()
  # plt.cla()
  # df = df[df['epoch'] == 30]
  # h = sns.ecdfplot(data=df, x='vals')
  # h.set(xlabel='Integer value of chromosome')
  # # ax.set_xlim(xmin = -10, xmax=1000)
  # # plt.axvline(x=32864, color='r')
  # plt.show()
  print(df.vals.value_counts())