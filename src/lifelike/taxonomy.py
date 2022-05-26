import csv

import numpy as np

from lifelike.CAs import CA
from util import binary

NUM_ITERS = 1000
NUM_EXPS = 100

if __name__ == "__main__":
    data = {
        "rstring": [],
        "conv_perc": [],
        "conv_mean": [],
        "conv_std": [],
        "period_mean": [],
        "period_std": []
    }
    for rstring in range(2**18):
        if not rstring % 2**10:
            with open("taxonomy.csv", "w") as outfile:
                writer = csv.writer(outfile)
                writer.writerow(data.keys())
                writer.writerows(zip(*data.values()))

        print(f'{rstring}/{2**18}')
        conv, period = [], []
        for exp in range(NUM_EXPS):
            b = binary.ones(rstring >> 9)
            s = binary.ones(rstring)
            history = dict()
            ca = CA.random(b, s)
            for counter in range(NUM_ITERS):
                key = ca.X.data.tobytes()
                if key in history:
                    conv.append(counter)
                    period.append(counter - history[key])
                    break
                history[key] = counter
                ca.simple_step()
        data["rstring"].append(rstring)
        data["conv_perc"].append(len(conv) / NUM_EXPS)
        data["conv_mean"].append(np.mean(conv))
        data["conv_std"].append(np.std(conv))
        data["period_mean"].append(np.mean(period))
        data["period_std"].append(np.std(period))

