import math
import random
import numpy as np

from gray_scott.CAs import MimicCA
from gray_scott.Chromosome import Chromosome


class Population:
    def __init__(self, n_parents, n_children, true_f, true_k, algorithm, recombination, selection, initialisation):
        self.n_parents = n_parents
        self.n_children = n_children
        self.pop_size = n_parents + n_children
        self.global_learning_rate = np.float_power(2, -0.5)
        self.local_learning_rate = np.float_power(2, -0.25)
        self.true_f = true_f
        self.true_k = true_k
        self.algorithm = algorithm
        self.recombination = recombination
        self.selection = selection
        self.real = MimicCA.empty(self.true_f, self.true_k)
        if initialisation == "THRESHOLD":
            self.inds = np.array([Chromosome.threshold(control=np.array([0.001, 0.001])) for _ in range(self.pop_size)])
        elif initialisation == "RANDOM":
            self.inds = np.array([Chromosome.random(control=np.array([0.001, 0.001])) for _ in range(self.pop_size)])
        else:
            raise Exception("Invalid argument for initialisation type")

    def iterate(self):
        # Crossover
        if self.algorithm == "ES":
            children = self.esx()
        elif self.algorithm == "GA":
            children = self.blx()
        else:
            raise Exception("Invalid argument for algorithm type")

        # Recombination
        if self.recombination == "PLUS":
            self.inds = np.append(self.inds, children)
        elif self.recombination == "COMMA":
            self.inds = children
        else:
            raise Exception("Invalid argument for recombination method")

        # Mutation (nothing for ES)
        if self.algorithm == "GA":
            for ind in self.inds:
                ind.state += np.random.normal(scale=ind.control)
                ind.state = ind.state.clip(min=0)

        # Selection
        loss = self.loss()
        if self.selection == "LINEAR":
            self.linear_rank_update(loss)
        elif self.selection == "ROULETTE":
            self.roulette_update(loss)
        else:
            raise Exception("Invalid argument for selection method")
        return self.evaluate(loss)

    def esx(self):
        # Evolutionary strategy crossover
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
        return np.array(children)

    def blx(self, alpha=0.1):
        # Blended alpha crossover
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
        return np.array(children)

    def linear_rank_update(self, loss):
        self.inds = self.inds[loss.argsort()]
        self.inds = self.inds[:self.n_parents]

    def roulette_update(self, loss):
        fitness = 1 - loss
        self.inds = np.random.choice(self.inds, size=self.n_parents,
                                     p=None if fitness.sum() == 0 else fitness / fitness.sum())

    def evaluate(self, loss):
        return np.mean(np.sort(loss)[:self.n_parents])

    def loss(self):
        return np.array([r.loss(self.real) for r in self.inds])
