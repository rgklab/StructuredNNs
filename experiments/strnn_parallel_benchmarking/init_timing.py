from time import time

import numpy as np

from strnn import StrNN

n_trials = 5
dimensions = [10, 50, 100, 250, 500, 750, 1000, 1500, 2000, 3000]
times = {d: [] for d in dimensions}

for d in dimensions:
    print(d)
    for _ in range(n_trials):
        s = time()
        A = np.random.uniform(0, 1, size=(d, d))
        A = A > 0.5

        StrNN(d, [d, d, d], d, opt_type="greedy", adjacency=A)
        e = time()
        print(e - s)
        times[d].append(e - s)

for k, v in times.items():
    print("Dim: {} - Avg Time: {}".format(k, np.mean(v)))
