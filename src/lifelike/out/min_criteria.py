import csv
import random

import numpy as np

from lifelike.CAs import CA, GRID_SIZE
from lifelike.constants import CHROMOSOME_LEN
from util import binary

NUM_ITERS = 5
NUM_EXPS = 10

if __name__ == "__main__":
    data = {
        "rstring": [],
        "negentropy": [],
    }
    seeds = [np.random.random((GRID_SIZE, GRID_SIZE)) > random.random() for _ in range(NUM_EXPS)]
    counter = 1
    for rstring in range((2**18) + 1):
        if not rstring % 2**12:
            print(f'{counter}/{2**6}')
            counter += 1
            with open("random_negentropy.csv", "w") as outfile:
                writer = csv.writer(outfile)
                writer.writerow(data.keys())
                writer.writerows(zip(*data.values()))

        negen = []
        for exp in range(NUM_EXPS):
            b = binary.ones(rstring >> CHROMOSOME_LEN // 2)
            s = binary.ones(rstring)
            ca = CA(seeds[exp], b, s)
            ca.step(steps=NUM_ITERS)
            p = np.mean(ca.X)
            negen.append(p * np.log2(p))
        data['rstring'].append(rstring)
        data["negentropy"].append(np.mean(negen))

