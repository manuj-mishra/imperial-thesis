import math
import random
import numpy as np

from gray_scott.CAs import MimicCA
from gray_scott.Chromosome import Chromosome


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

    def ca_crossover(self, alpha=0.1):
        # BLX-alpha crossover
        children = []
        for _ in range(self.n_children):
            parents = np.random.choice(self.inds, 2, replace=False)
            a, b = parents[0], parents[1]
            af, ak = a.state[0], a.state[1]
            bf, bk = b.state[0], b.state[1]
            adf, adk = a.control[0], a.control[1]
            bdf, bdk = b.control[0], b.control[1]
            deltF = abs(af - bf)
            deltK = abs(ak - bk)
            child = Chromosome(
                state=np.array([af if random.random() < 0.5 else bf,
                                ak if random.random() < 0.5 else bk]),
                control=np.array([np.random.uniform(low=max(min(adf, bdf) - alpha * deltF, 0),
                                                    high=max(adf, bdf) + alpha * deltF),
                                  np.random.uniform(low=max(min(adk, bdk) - alpha * deltK, 0),
                                                    high=max(adk, bdk) + alpha * deltK)])
            )
            children.append(child)
        self.inds = np.append(self.inds, np.array(children))

    def update(self, loss):
        self.inds = self.inds[loss.argsort()]
        self.inds = self.inds[:self.n_parents]

    def evaluate(self, loss):
        return np.mean(np.sort(loss)[:self.n_parents])

    def loss(self):
        true = MimicCA.empty(self.true_f, self.true_k)
        return np.array([r.loss(true) for r in self.inds])
