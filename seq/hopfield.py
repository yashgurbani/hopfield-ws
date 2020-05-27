#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: yash
"""

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from random import choice
import datetime
from tqdm import tqdm
import random

n = int(input('Enter # of neurons'))
flip = round(0.25 * n)
n_i = 10000 #int(input('Enter # of iterations each run'))
ensembleCount = 1 #int(input('Enter # of runs to average over'))
ErrorMatrixn = np.zeros((ensembleCount, 200))  # initiate quality matrix

for b in (range(ensembleCount)):
    Y = []
    X = []
    hammingtemp = 0

    for counter in tqdm(range (1, 201)):  # for this given random matrix, loop over different p values
        alpha = counter/1000
        m = round(alpha * n)
        MemoryMatrix = np.zeros((m, n))
        target_list = []

        for a in range(0, m):
            mem = []
            for i in range(0, n):
                x = choice([1, -1])
                mem.append(x)
            target_list.append(a)
            MemoryMatrix[a] = mem
            x = 0

        target = random.choice(target_list)

        u = MemoryMatrix[target].copy()  # copy target M to initial state

        n_init = random.randint(0, (n - flip))  # start at a random position in the initial state
        n_fin = n_init + flip

        for count in range(n_init, n_fin):  # sequentially pick up 25% of bits and flip them
            u[count] = (MemoryMatrix[target][count] * -1)

        g = nx.complete_graph(n)

        # set the initial state
        st = 0
        for st in range(0, n):
            g.nodes[st]['state'] = u[st]

        # train the network
        for i, j in g.edges:
            weight = 0
            zeta = 0
            for zeta in range(0, m):
                weight = weight + (MemoryMatrix[zeta][i] * MemoryMatrix[zeta][j])
            g.edges[i, j]['weight'] = (weight / n)

        # evolve according to hopfield dynamics, n_i iterations
        for z in range(0, n_i):
            i = choice(list(g.nodes))
            s = sum([g.edges[i, j]['weight'] * g.nodes[j]['state'] for j in g.neighbors(i)])
            g.nodes[i]['state'] = 1 if s > 0 else -1 if s < 0 else g.nodes[i]['state']

            # nx.draw(g, pos = g.pos, cmap = cm.jet, vmin = -1, vmax = 1, node_color = [g.nodes[i]['state'] for i in
            # g.nodes]) plt.show()
            z = z + 1

        # calculate hamming distances

        for i in list(g.nodes):
            hammingtemp += abs(g.nodes[i]['state'] - MemoryMatrix[target][i])

        X.append(alpha)
        Y.append(hammingtemp)


    ErrorMatrixn[b] = Y
    b = b + 1

col_totalsE = [sum(x) for x in zip(*ErrorMatrixn)]
col_totalsEavg = [(x / 2 * n * ensembleCount) for x in col_totalsE]

# set up figure and axes
plt.plot(X, col_totalsEavg, color="magenta")
plt.ylabel("% of errors")
plt.xlabel("alpha")

suffix = datetime.datetime.now().strftime("%m%d_%H%M%S")
filename = "_".join([str(n), suffix])
plt.autoscale()
plt.savefig(filename, dpi=200, bbox_inches = "tight")
plt.show()

np.savetxt(filename + "-s-Q.csv", np.column_stack((X, col_totalsEavg)), delimiter=",", fmt='%s')
