#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: yash
"""

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from random import choice
from matplotlib import cm
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

flip = round(0.25 * n)
n_i = 50

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

g.pos = {}
for x in range(4):
   for y in range(4):
       g.pos[y * 4 + x] = (x, -y)


nx.draw(g, pos = g.pos, cmap = cm.jet, vmin = -1, vmax = 1, node_color = [g.nodes[i]['state'] for i in
g.nodes])
plt.show()
print ("Initial Graph")


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

# evolve according to hopfield dynamics, n_i iterations
for z in range(0, n_i):
    i = choice(list(g.nodes))
    s = sum([g.edges[i, j]['weight'] * g.nodes[j]['state'] for j in g.neighbors(i)])
    g.nodes[i]['state'] = 1 if s > 0 else -1 if s < 0 else g.nodes[i]['state']
    for q in range(m):
        for i in list(g.nodes):
            hamming_distance [z, q] += (abs(g.nodes[i]['state'] - MemoryMatrix[q][i])/2)
    z = z + 1

nx.draw(g, pos = g.pos, cmap = cm.jet, vmin = -1, vmax = 1, node_color = [g.nodes[i]['state'] for i in g.nodes])
plt.show()
fig = plt.figure(figsize = (8, 8))
plt.plot(hamming_distance)
plt.xlabel('No of Iterations')
plt.ylabel('Hamming Distance')
plt.ylim([0, n])
plt.plot(hamming_distance)

#plt.legend(['pattern 0', 'pattern 1', 'pattern 2', 'pattern 3', 'pattern 4', 'pattern 5', 'pattern 6', 'pattern 7', 'pattern 8', 'pattern 9', 'pattern 10', 'pattern 11', 'pattern 12', 'pattern 13', 'pattern 14', 'pattern15', 'pattern16'], loc='best')
plt.show()




