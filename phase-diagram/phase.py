import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from random import choice
import datetime
from tqdm import tqdm
import random

type = int(input('Enter network type: 1 for a fully connected network, 2 for a watts-strogatz network'))
n = int(input('Enter # of neurons'))

if (type == 2):
 k = int(input('Enter # of nearest neighbours k'))
 p = float(input('Enter probability of rewiring p'))

m_f = round(0.2 * n)
MemoryMatrix = np.zeros((m_f, n))
ensembleCount = int(input('Enter # of ensembles to average over'))

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

OverlapMatrixn = np.zeros((ensembleCount, round((m_f-1)/steps)))  # initiate quality matrix

for b in tqdm(range(0, ensembleCount)):
    target = random.randint(0, 2)
    Y = []
    X = []
    hamming_distance = np.zeros((n_i, 3))
    u = MemoryMatrix[target].copy()  # copy target M to initial state
    rs = random.sample(range(0, n), flip)

    for z in list(rs):  # randomly pick up 25 percent and flip them
        u[z] = (MemoryMatrix[target][z] * -1)
        
        
    if (type==1):
     g = nx.complete_graph(n)
    
    if (type==2):
     g = nx.watts_strogatz_graph(n,k,p) #use for watts strogatz network

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
                    hamming_distance [z, target] += (abs(g.nodes[i]['state'] - MemoryMatrix[target][i])/2)
                    
            if (hamming_distance [z, 0] == 0):
                break
                    
            z = z + 1

        # calculate overlaps
        overlaptemp = 0
        for i in list(g.nodes):
            overlaptemp += (g.nodes[i]['state'] * MemoryMatrix[target][i])

        X.append(alpha)
        Y.append(overlaptemp)
        
    OverlapMatrixn[b] = Y

col_totalsO = [sum(x) for x in zip(*OverlapMatrixn)]
col_totalsOavg = [(x / (n*ensembleCount)) for x in col_totalsO]

# set up fiagure and axes
plt.plot(X, col_totalsOavg, color="magenta")
plt.ylabel("overlap with the target state")
plt.xlabel("alpha")
suffix = datetime.datetime.now().strftime("%m%d_%H%M%S")

if (type==1):
 filename = "_".join([str(type), str(n), str(k), str(p), suffix])

if (type==2):
 filename = "_".join([str(type), str(n), suffix])

plt.autoscale()
plt.savefig(filename, dpi=200, bbox_inches = "tight")
plt.show()

np.savetxt(filename + "-s-Q.csv", np.column_stack((X, col_totalsOavg)), delimiter=",", fmt='%s')
