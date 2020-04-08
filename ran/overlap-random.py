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

#yes it works
n = int(input('Enter # of neurons'))
m = int(input('Enter # of memory states'))
flip = round(0.25 * n)

MemoryMatrix = np.zeros((m, n))

for a in range(0, m):
    mem = []
    for i in range(0, n):
        x = choice([1, -1])
        mem.append(x)

    MemoryMatrix[a] = mem
    x = 0

k = int(input('Enter # of connected neighbours k'))
p = 0
n_i = int(input('Enter # of iterations each run'))
ensembleCount = int(input('Enter # of runs to average over'))
target = int(input('Select memory state which when added noise gives initial state'))

suffix = datetime.datetime.now().strftime("%m%d_%H%M%S")
filename = "_".join([str(n), str(m), str(k), str(n_i), str(ensembleCount), suffix])
np.savetxt(filename + "-ms-s.csv", np.column_stack((np.transpose(MemoryMatrix))), delimiter=",", fmt='%s')

OverlapMatrixn = np.zeros((ensembleCount, n))  # initiate overlap matrix

for b in tqdm(range(ensembleCount)):
    u = []
    u = MemoryMatrix[target].copy()  # copy target M to initial state

    Y = []
    X = []

    n_init = 0
    n_fin = n_init + flip

    for count in range(n_init, n_fin):  # sequentially pick up 25% of bits and flip them
        u[count] = (MemoryMatrix[target][count] * -1)

    g = nx.watts_strogatz_graph(n, k, p)

    st = 0
    for st in range(0, n):
        g.nodes[st]['state'] = u[st]
        X.append(st)


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
     isCorrect = 0
     if (MemoryMatrix[target][i] == g.nodes[i]['state']):
          isCorrect = 1
     Y.append(isCorrect)

    OverlapMatrixn[b] = Y
    b = b + 1

totalCorrect = [sum(x) for x in zip(*OverlapMatrixn)]
tCorrect =  [(x / ensembleCount) for x in totalCorrect]
# set up figure and axes
plt.plot(X, tCorrect, color="black")
plt.ylabel("probability of correctness")
plt.xlabel("neuron")
suffix = datetime.datetime.now().strftime("%m%d_%H%M%S")
filename = "_".join([str(n_i), str(ensembleCount), suffix])
plt.autoscale()
plt.savefig(filename, dpi=200, bbox_inches = "tight")
plt.show()

np.savetxt(filename + "-tcorrect.csv", np.column_stack((X, tCorrect)), delimiter=",", fmt='%s')

