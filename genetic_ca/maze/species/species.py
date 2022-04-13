import math
import random

import pandas as pd
import networkx as nx
from pyvis.network import Network

def graph(n=0):
  df = pd.read_csv(f'./out/bias{n}.0_pop50_el50.0_mut5.0/elite.csv')
  df = df.iloc[0].astype(int)
  to_bin = lambda x: int(str(x), 2)
  df = df.apply(to_bin)
  G = nx.Graph()
  for i in df:
    G.add_node(i)
    for j in df:
      n = same_bit_n(i, j)
      if 13 < n < 16:
        G.add_edge(i, j, weight=n / 16)
  return G


def same_bit_n(a, b):
  # Returns number of bits that are the same in a and b
  c = a ^ b
  count = 16
  while c:
    c &= c - 1
    count -= 1
  return count

if __name__ == "__main__":
  for i in range(6):
    n = 20 * i
    G = graph(n)
    print(G)
    net = Network(notebook=True)
    net.from_nx(G)
    net.show(f'test{n}.html')

