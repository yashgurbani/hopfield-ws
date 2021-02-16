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

type = int(input('Enter network type: 1 for a fully connected network, 2 for a watts-strogatz network'))

n = int(input('Enter # of neurons'))
m = int(input('Enter # of memory states'))

if (type == 2):
 k = int(input('Enter # of nearest neighbours k'))
 p = float(input('Enter probability of rewiring p'))

n_i = int(input('Enter # of iterations'))
MemoryMatrix = np.zeros((m, n))
W = np.zeros((n, n))

for a in range(0, m):
    mem = []
    for i in range(0, n):
        x = choice([1, -1])
        mem.append(x)
    MemoryMatrix[a] = mem
    x = 0

print(MemoryMatrix)
flip = round(0.25 * n)


Y = []
X = []
u = []
u = MemoryMatrix[1].copy()  # copy target M to initial state
rs = random.sample(range(0, n), flip)

for z in list(rs):  # randomly pick up 25 percent and flip them
    u[z] = (MemoryMatrix[1][z] * -1)

if (type == 1):
        g = nx.complete_graph(n)
if (type == 2):
        g = nx.watts_strogatz_graph(n, k, p)  # use for watts strogatz network

st = 0
for st in range(0, n):
    g.nodes[st]['state'] = u[st]

alpha = (m/n)

if (n == 16):

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

if (n == 16):
    nx.draw(g, pos = g.pos, cmap = cm.jet, vmin = -1, vmax = 1, node_color = [g.nodes[i]['state'] for i in g.nodes])
    plt.show()
    print ("Final Graph")

fig = plt.figure(figsize = (8, 8))
plt.plot(hamming_distance)
plt.xlabel('No of Iterations')
plt.ylabel('Hamming Distance')
plt.ylim([0, n])
plt.plot(hamming_distance)
plt.show()




