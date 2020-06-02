import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from random import choice
import datetime
from tqdm import tqdm
import random

n = int(input('Enter # of neurons'))
m_f = round(0.2 * n)
MemoryMatrix = np.zeros((m_f, n))
ensembleCount = 10

for a in range(0, m_f):
    mem = []
    for i in range(0, n):
        x = choice([1, -1])
        mem.append(x)
    MemoryMatrix[a] = mem
    x = 0

flip = round(0.2 * n)
n_i = 10000
steps = int(input('Enter # of steps in iteration of m (resolution)'))

ErrorMatrixn = np.zeros((1, round((m_f-1)/2)))  # initiate quality matrix

for b in tqdm(range(0, ensembleCount)):
    

    Y = []
    X = []
    hamming_distance = np.zeros((n_i, 1))
    u = MemoryMatrix[1].copy()  # copy target M to initial state
    rs = random.sample(range(0, n), flip)

    for z in list(rs):  # randomly pick up 25 percent and flip them
        u[z] = (MemoryMatrix[1][z] * -1)

    g = nx.complete_graph(n)

    st = 0
    for st in range(0, n):
        g.nodes[st]['state'] = u[st]


    for m in (range (2, (m_f+1), steps)):  # for this given random matrix, loop over different m values
        alpha = (m/n)

        # train the network with m memory states
        for i, j in g.edges:
            weight = 0
            for zeta in range(0, m):
                weight = weight + (MemoryMatrix[zeta][i] * MemoryMatrix[zeta][j])
            g.edges[i, j]['weight'] = (weight / n)

        # evolve according to hopfield dynamics, n_i iterations
        for z in range(0, n_i):
            i = choice(list(g.nodes))
            s = sum([g.edges[i, j]['weight'] * g.nodes[j]['state'] for j in g.neighbors(i)])
            g.nodes[i]['state'] = 1 if s > 0 else -1 if s < 0 else g.nodes[i]['state']
            for i in list(g.nodes):
                    hamming_distance [z, 0] += (abs(g.nodes[i]['state'] - MemoryMatrix[1][i])/2)
                    
            if (hamming_distance [z, 0] == 0):
                break
                    
            z = z + 1

        # calculate overlaps
        overlaptemp = 0
        for i in list(g.nodes):
            overlaptemp += (g.nodes[i]['state'] * MemoryMatrix[1][i])

        X.append(alpha)
        Y.append(hammingtemp)


col_totalsO = [sum(x) for x in zip(*EtaMatrixn)]
col_totalsOavg = [(x / ensembleCount) for x in col_totalsE]

# set up figure and axes
plt.plot(X, col_totalsOavg, color="magenta")
plt.ylabel("overlap with the target state")
plt.xlabel("alpha")
png = ".png"
suffix = datetime.datetime.now().strftime("%m%d_%H%M%S")
filename = "_".join([str(n), suffix])
plt.autoscale()
plt.savefig(filename, dpi=200, bbox_inches = "tight")
plt.show()

np.savetxt(filename + "-s-Q.csv", np.column_stack((X, col_totalsEavg)), delimiter=",", fmt='%s')
