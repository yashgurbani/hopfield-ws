#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: yash
"""

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from random import choice
from matplotlib import cm
import datetime
from tqdm import tqdm
from random import randrange
import random

n = int(input('Enter # of neurons'))
m = int(input('Enter # of memory states'))
flip = round(0.25 * n)

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

n_i = 10000 #int(input('Enter # of iterations each run'))
ensembleCount = int(input('Enter # of runs to average over'))
#target = int(input('Select memory state which when added noise gives initial state'))

suffix = datetime.datetime.now().strftime("%m%d_%H%M%S")
filename = "_".join([str(n), str(m), str(n_i), str(ensembleCount), suffix])
np.savetxt(filename + "-ms-s.csv", np.column_stack((np.transpose(MemoryMatrix))), delimiter=",", fmt='%s')

OverlapMatrixn = np.zeros((ensembleCount, 50 ))  # initiate overlap matrix

for b in tqdm(range(ensembleCount)):
    target = random.choice(target_list)
    u = []
    u = MemoryMatrix[target].copy()  # copy target M to initial state
    hammingtemp = overlaptemp = 0
    hammingavg = overlapavg = 0
    Y = []
    X = []

    rho = 0
    p = 0


    rs = random.sample(range(0, n), flip)

    for p in list(rs):  # randomly pick up 25 percent and flip them
        u[p] = (MemoryMatrix[target][p] * -1)

    for k in (range(1,51)):  # for this given random matrix, loop over different p values

        hammingtemp = overlaptemp = 0
        hammingval = overlapval = 0
        q = 0

        g = nx.barabasi_albert_graph(n, k, seed=None)

        # set the initial state
        st = 0
        for st in range(0, n):
            g.nodes[st]['state'] = u[st]

        # nx.draw(g, pos = g.pos, cmap = cm.jet, vmin = -1, vmax = 1, node_color = [g.nodes[i]['state'] for i in
        # g.nodes]) plt.show() print ("Initial Graph")

        # train the network
        for i, j in g.edges:
            weight = 0
            alpha = 0
            for alpha in range(0, m):
                weight = weight + (MemoryMatrix[alpha][i] * MemoryMatrix[alpha][j])
            g.edges[i, j]['weight'] = (weight / n)

        # evolve according to hopfield dynamics, n_i iterations
        for z in range(0, n_i):
            i = choice(list(g.nodes))
            s = sum([g.edges[i, j]['weight'] * g.nodes[j]['state'] for j in g.neighbors(i)])
            g.nodes[i]['state'] = 1 if s > 0 else -1 if s < 0 else g.nodes[i]['state']

            # nx.draw(g, pos = g.pos, cmap = cm.jet, vmin = -1, vmax = 1, node_color = [g.nodes[i]['state'] for i in
            # g.nodes]) plt.show()
            z = z + 1

        # calculate hamming distances and overlap for all the memory states

        for i in list(g.nodes):
            overlaptemp += (MemoryMatrix[target][i] * g.nodes[i]['state'])

        overlapval = (overlaptemp / n)

        X.append(k)
        Y.append(overlapval)

    OverlapMatrixn[b] = Y

    b = b + 1

col_totalsO = [sum(x) for x in zip(*OverlapMatrixn)]
col_totalsOavg = [(x / ensembleCount) for x in col_totalsO]

# set up figure and axes
plt.plot(X, col_totalsOavg, color="magenta")
plt.title("\n Variation of Network Performance with Degree Distribution \n %s neruons with %s memory states \n "
          "Iterations = %s | Ensemble Count= ""%s \n \n" % (n, m, n_i, ensembleCount))
plt.ylabel("average quality q")


suffix = datetime.datetime.now().strftime("%m%d_%H%M%S")
filename = "_".join([str(n), str(m), str(k), str(n_i), str(ensembleCount), suffix])
plt.autoscale()
plt.savefig(filename, dpi=200, bbox_inches = "tight")
plt.show()

np.savetxt(filename + "-s-O.csv", np.column_stack((X, col_totalsOavg)), delimiter=",", fmt='%s')
