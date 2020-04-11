#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: yash
"""
# change
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import networkx as nx
from random import choice
from matplotlib import cm
import datetime
from tqdm import tqdm
from random import randrange
import random

div = 10
n = int(input('Enter # of neurons'))
m = int(input('Enter # of memory states'))

k = int(input('Enter # of connected neighbours k'))
n_i = int(input('Enter # of iterations each run'))
ensembleCount = int(input('Enter # of realizations'))
EfficacyMatrix = np.zeros((ensembleCount, div))  # initiate quality matrix

for b in tqdm(range(ensembleCount)): #realizations

    MemoryMatrix = np.zeros((m, n))
    MemoryMatrixR = np.zeros((m, n))
    for a in range(0, m): #random memory states
        mem = []
        for i in range(0, n):
            x = choice([1, -1])
            mem.append(x)
        res = np.flipud(mem)
        MemoryMatrix[a] = mem
        MemoryMatrixR[a] = res
        x = 0

    u =  []
    for a in range(0, n): #randomized init state
       x = choice([1, -1])
       u.append(x)
       x = 0

    hammingtemp = overlaptemp = 0
    hammingavg = overlapavg = 0
    Y = []
    Y2 = []
    X = []

    for rho in (range(0, div+1)):  # for this given random matrix, loop over different p values

        p = (rho / div)
        hammingtemp = overlaptemp = 0
        hammingval = overlapval = 0
        q = 0

        g = nx.watts_strogatz_graph(n, k, p)

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
        finalstate = []
        for i in list(g.nodes):
            g.nodes[i]['state'] = x
            finalstate.append(x)
            x = 0

        eCount = 0
        for a in range (0, m):
            if (np.array_equal(finalstate, MemoryMatrix[a])):
                eCount = eCount + 1
            if (np.array_equal(finalstate, MemoryMatrixR[a])):
                eCount = eCount + 1
        X.append(p)
        Y.append(eCount)

    EfficacyMatrix[b] = Y

    b = b + 1

col_totalsEff = np.sum(EfficacyMatrix, 0)
col_totalsEffavg = [(x / ensembleCount) for x in col_totalsEff]

# set up figure and axes
plt.plot(X, col_totalsEffavg, color="magenta")
plt.title("\n Variation of Network Performance with Rewiring Probability \n %s neurons with %s memory states \n "
          "Iterations = %s |  k = %s | Ensemble Count= ""%s \n \n" % (n, m, n_i, k, ensembleCount))
plt.ylabel("efficacy")

eff = "EFF"
suffix = datetime.datetime.now().strftime("%m%d_%H%M%S")
filename = "_".join([eff, str(n), str(m), str(k), str(n_i), str(ensembleCount), suffix])
plt.autoscale()
plt.savefig(filename, dpi=200, bbox_inches="tight")
plt.show()

np.savetxt(filename + "-r-Eff.csv", np.column_stack((X, col_totalsEffavg)), delimiter=",", fmt='%s')
