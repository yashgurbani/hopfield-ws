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
m = int(input('Enter # of memory states'))

MemoryMatrix = np.zeros((m, n))
W = np.zeros((n, n))

for a in range(0, m):
    mem = []
    for i in range(0, n):
        x = choice([1, -1])
        mem.append(x)
    MemoryMatrix[a] = mem
    x = 0

flip = round(0.3 * n)
n_i = 10000

ErrorMatrixn = np.zeros((1, round((m-1)/2)))  # initiate quality matrix

Y = []
X = []
u = []
u = MemoryMatrix[1].copy()  # copy target M to initial state
rs = random.sample(range(0, n), flip)

for z in list(rs):  # randomly pick up 25 percent and flip them
    u[z] = (MemoryMatrix[1][z] * -1)

g = nx.complete_graph(n)

st = 0
for st in range(0, n):
    g.nodes[st]['state'] = u[st]

alpha = (m/n)

# train the network with m memory states
for i, j in g.edges:
    weight = 0
    for zeta in range(0, m):
        weight = weight + (MemoryMatrix[zeta][i] * MemoryMatrix[zeta][j])
    g.edges[i, j]['weight'] = (weight / n)
    W[i][j] = g.edges[i, j]['weight']
    W[j][i] = W[i][j]

plt.imshow(W, cmap='gray')
plt.show()
hamming_distance = np.zeros((n_i, m))

z_index = []
# evolve according to hopfield dynamics, n_i iterations
for z in range(0, n_i):
    i = choice(list(g.nodes))
    s = sum([g.edges[i, j]['weight'] * g.nodes[j]['state'] for j in g.neighbors(i)])
    g.nodes[i]['state'] = 1 if s > 0 else -1 if s < 0 else g.nodes[i]['state']
    for q in range(m):
        for i in list(g.nodes):
            hamming_distance [z, q] += (abs(g.nodes[i]['state'] - MemoryMatrix[q][i])/2)
    z_index.append(z)
    z = z + 1


fig = plt.figure(figsize = (8, 8))
plt.plot(hamming_distance)
plt.xlabel('No of Iterations')
plt.ylabel('Hamming Distance')
plt.ylim([0, 400])
c = "pattern"
legend = ','.join("'%s'" % '{} {}'.format(c, i) for i in range(0, m))
plt.legend([legend], loc='best')
plt.show()


