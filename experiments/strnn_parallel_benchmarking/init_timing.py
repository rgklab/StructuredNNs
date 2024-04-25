from time import time

import numpy as np

from strnn import StrNN

n_trials = 5
expansion = 5
n_layer = 3
dimensions = [10, 50, 100, 250, 500, 750, 1000, 1500, 2000, 3000]

times: dict[int, list[float]] = {d: [] for d in dimensions}

for d in dimensions:
    print(d)
    for _ in range(n_trials):
        s = time()
        A_samp = np.random.uniform(0, 1, size=(d, d))
        A = A_samp > 0.5

        n_hidden = tuple([d * expansion] * n_layer)
        StrNN(d, n_hidden, d, opt_type="greedy_parallel", adjacency=A)
        e = time()
        print(str(e - s) + ",")
        times[d].append(e - s)

for k, v in times.items():
    print("Dim: {} - Avg Time: {}".format(k, np.mean(v)))
