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
flip = round(0.25 * n)
m_up = round(0.3 * n)
div = m_up
p = (input('Enter probability of rewiring p'))
n_i = int(input('Enter # of iterations each run'))
ensembleCount = int(input('Enter # of runs to average over'))


OverlapMatrixn = np.zeros((ensembleCount, div))  # initiate overlap matrix
EtaMatrixn = np.zeros((ensembleCount, div))  # initiate quality matrix

for b in tqdm(range(ensembleCount)):

    MemoryMatrix = np.zeros((m_up, n))

    for a in range(0, m_up):
        mem = []
    for i in range(0, n):
        x = choice([1, -1])
    mem.append(x)
    MemoryMatrix[a] = mem

    u = []
    u = MemoryMatrix[1].copy()  # copy M1 to initial state
    hammingtemp = overlaptemp = 0
    hammingavg = overlapavg = 0
    Y = []
    Y2 = []
    X = []

    n_init = random.randint(0, (n - flip))  # start at a random position in the initial state
    n_fin = n_init + flip

    for count in range(n_init, n_fin):  # sequentially pick up 25% of bits and flip them
        u[count] = (MemoryMatrix[1][count] * -1)


    for m in range(1, (m_up+1)):  # for this given random matrix, loop over different p values

        x = 0
        hammingtemp = overlaptemp = 0
        hammingval = overlapval = 0

        q = 0
        k = round(0.15*n)
        g = nx.watts_strogatz_graph(n, k , 0.4)

        # g.pos = {}
        # for x in range(4):
        #      for y in range(4):
        #         g.pos[y * 4 + x] = (x, -y)

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
        z = 0
        for z in range(1, (n_i + 1)):
            i = choice(list(g.nodes))
            s = sum([g.edges[i, j]['weight'] * g.nodes[j]['state'] for j in g.neighbors(i)])
            g.nodes[i]['state'] = 1 if s > 0 else -1 if s < 0 else g.nodes[i]['state']

            # nx.draw(g, pos = g.pos, cmap = cm.jet, vmin = -1, vmax = 1, node_color = [g.nodes[i]['state'] for i in
            # g.nodes]) plt.show()
            z = z + 1

        # calculate hamming distances and overlap for all the memory states
        i = v = 0
        for v in range(0, m):
            for i in list(g.nodes):
                hammingtemp += abs(g.nodes[i]['state'] - MemoryMatrix[v][i])
                overlaptemp += (MemoryMatrix[v][i] * g.nodes[i]['state'])

        hammingval = (hammingtemp / (m * 2))
        overlapval = (overlaptemp / (n * m))

        q = ((n - hammingval) / n)
        X.append(m)
        Y.append(q)
        Y2.append(overlapval)


    OverlapMatrixn[b] = Y2
    EtaMatrixn[b] = Y

    b = b + 1

col_totalsE = [sum(x) for x in zip(*EtaMatrixn)]
col_totalsO = [sum(x) for x in zip(*OverlapMatrixn)]

col_totalsEavg = [(x / ensembleCount) for x in col_totalsE]
col_totalsOavg = [(x / ensembleCount) for x in col_totalsO]

# set up figure and axes
plt.subplot(2, 1, 1)
plt.plot(X, col_totalsEavg, color="black")
plt.title("\n Variation of Network Performance with Memory Size\n %s neurons with %s nearest neighbours \n "
          "Iterations = %s | Ensemble Count= %s \n \n" % (n, k, n_i, ensembleCount))
plt.ylabel("average quality q")

plt.subplot(2, 1, 2)
plt.ylabel("average overlap m")
plt.xlabel("no of memory states m \n")
plt.plot(X, col_totalsOavg, color="purple")

suffix = datetime.datetime.now().strftime("%m%d_%H%M%S")
filename = "_".join([str(n), str(p), str(k), str(n_i), str(ensembleCount), suffix, ".png"])
plt.autoscale()
plt.savefig(filename, dpi=200, bbox_inches = "tight")
plt.show()

np.savetxt(filename + "-s-Q.csv", np.column_stack((X, col_totalsEavg)), delimiter=",", fmt='%s')
np.savetxt(filename + "-s-O.csv", np.column_stack((X, col_totalsOavg)), delimiter=",", fmt='%s')
