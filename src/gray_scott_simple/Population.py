import math
import random
import numpy as np

from gray_scott_simple.CAs import MimicCA
from gray_scott_simple.Chromosome import Chromosome



class Population:
  def __init__(self, n_parents, n_children, true_f, true_k):
    self.n_parents = n_parents
    self.n_children = n_children
    self.pop_size = n_parents + n_children
    self.global_learning_rate = np.float_power(2, -0.5)
    self.local_learning_rate = np.float_power(2, -0.25)
    self.true_f = true_f
    self.true_k = true_k
    self.inds = np.array([Chromosome.random(control=np.array([0.001, 0.001])) for _ in range(self.pop_size)])

  def iterate(self):
    children = []
    centroid_state = np.mean(np.array([c.state for c in self.inds]), axis=0)
    centroid_control = np.mean(np.array([c.control for c in self.inds]), axis=0)
    for i in range(self.n_children):
      global_step_size = self.global_learning_rate * np.random.standard_normal()
      local_step_size = self.local_learning_rate * np.random.standard_normal(size=2)
      mutation = np.random.standard_normal(size=2)
      new_control = np.exp(global_step_size) * np.multiply(centroid_control, np.exp(local_step_size))
      new_state = centroid_state + np.multiply(new_control, mutation)
      children.append(Chromosome(new_state, new_control))
    self.inds = np.append(self.inds, children)
    loss = self.loss()
    self.update(loss)
    return self.evaluate(loss)

  def update(self, loss):
    self.inds = self.inds[loss.argsort()]
    self.inds = self.inds[:self.n_parents]

  def evaluate(self, loss):
    return 1 - np.mean(np.sort(loss)[:self.n_parents])

  def loss(self):
    true = MimicCA.empty(self.true_f, self.true_k)
    return np.array([r.loss(true) for r in self.inds])
