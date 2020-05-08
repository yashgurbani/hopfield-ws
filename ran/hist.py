#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: yash
"""
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

n = int(input('Enter # of neurons'))
m = int(input('Enter # of memory states'))
flip = round(0.25 * n)
MemoryMatrix = np.zeros((m, n))

for a in range(0, m): #generate memory matrix
    mem = []
    for i in range(0, n): #each row is a configuration state for n neurons
        x = choice([1, -1])
        mem.append(x)

    MemoryMatrix[a] = mem
    x = 0

k = int(input('Enter # of connected neighbours k'))
n_i = int(input('Enter # of iterations each run'))
ensembleCount = int(input('Enter # of runs to average over'))
target = int(input('Select memory state which when added noise gives initial state'))

suffix = datetime.datetime.now().strftime("%m%d_%H%M%S")
filename = "_".join([str(n), str(m), str(k), str(n_i), str(ensembleCount), suffix])
#np.savetxt(filename + "-ms-r.csv", np.column_stack((np.transpose(MemoryMatrix))), delimiter=",", fmt='%s')

div = 10 #resolution
OverlapMatrixn = np.zeros((ensembleCount, div))  # initiate overlap matrix
EtaMatrixn = np.zeros((ensembleCount, div))  # initiate quality matrix

for b in tqdm(range(ensembleCount)): #loop over various init configs
    u = []
    u = MemoryMatrix[target].copy()  # copy target to initial test pattern
    hammingtemp = overlaptemp = 0
    hammingavg = overlapavg = 0
    Y = []
    Y2 = []
    X = []
    p = 0

    rs = random.sample(range(0, n), flip)

    for p in list(rs):  # randomly pick up 25 percent and flip them
        u[p] = (MemoryMatrix[target][p] * -1)

    for rho in (range(0, 10)):  # for this given random matrix, loop over different p values

        p = (rho / div)
        hammingtemp = overlaptemp = 0
        hammingval = overlapval = 0
        q = 0

        g = nx.watts_strogatz_graph(n, k, p) #generate the graph

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
            i = choice(list(g.nodes)) #pick up a node randomly
            s = sum([g.edges[i, j]['weight'] * g.nodes[j]['state'] for j in g.neighbors(i)])
            g.nodes[i]['state'] = 1 if s > 0 else -1 if s < 0 else g.nodes[i]['state']

            # nx.draw(g, pos = g.pos, cmap = cm.jet, vmin = -1, vmax = 1, node_color = [g.nodes[i]['state'] for i in
            # g.nodes]) plt.show()
            z = z + 1
            # calculate hamming distances and overlap for all the memory states
        weights_hist = []
        binwidth = 0.001
        for i, j in g.edges:
            tempw = g.edges[i, j]['weight']
            weights_hist.append(tempw)
        hist = "HIST"
        png = ".png"
        filename = "_".join([hist, str(p), str(n), str(m), png])
        plt.clf()
        plt.hist(weights_hist, bins=np.arange(min(weights_hist), max(weights_hist) + binwidth, binwidth))
        plt.savefig(filename, dpi=200, bbox_inches="tight")

        for i in list(g.nodes):
            overlaptemp += (MemoryMatrix[target][i] * g.nodes[i]['state'])
            hammingtemp += abs(g.nodes[i]['state'] - MemoryMatrix[target][i])

        hammingval = (hammingtemp / 2)
        overlapval = (overlaptemp / n)
        q = ((n - hammingval) / n)
        X.append(p)
        Y.append(q)
        Y2.append(overlapval)

    OverlapMatrixn[b] = Y2
    EtaMatrixn[b] = Y

    b = b + 1

col_totalsE = np.sum(EtaMatrixn, 0)
col_totalsO = np.sum(OverlapMatrixn, 0)

col_totalsEavg = [(x / ensembleCount) for x in col_totalsE]
col_totalsOavg = [(x / ensembleCount) for x in col_totalsO]




#np.savetxt(filename + "-r-Qu.csv", np.column_stack((X, col_totalsEavg)), delimiter=",", fmt='%s')
#np.savetxt(filename + "-r-Ov.csv", np.column_stack((X, col_totalsOavg)), delimiter=",", fmt='%s')
