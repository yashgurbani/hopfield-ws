import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from random import choice
import datetime
from tqdm import tqdm
import random

np.set_printoptions(threshold=sys.maxsize)

os.chdir(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_excel('s.xlsx', header=None, skiprows =1, usecols=range(1,515), sheet_name="male chem synapse adjacency")
c = df.to_numpy()

c[c > 0] = 1

n = 514
m_f = round(0.2 * n)
MemoryMatrix = np.zeros((m_f, n))


for a in range(0, m_f):
    mem = []
    for i in range(0, n):
        x = choice([1, -1])
        mem.append(x)
    MemoryMatrix[a] = mem
    x = 0

flip = round(0.3 * n)
n_i = 10000

W = np.zeros((n, n))

Y = []
X = []
u = []
u = MemoryMatrix[1].copy()  # copy target M to initial state
rs = random.sample(range(0, n), flip)
ErrorMatrixn = np.zeros((1, round((m_f-1)/2)))  # initiate quality matrix
steps = int(input('Enter # of steps in iteration of m (resolution)'))

for z in list(rs):  # randomly pick up 25 percent and flip them
    u[z] = (MemoryMatrix[1][z] * -1)

for m in tqdm(range (2, (m_f+1), steps)):  # for this given random matrix, loop over different m values

    alpha = (m/n)

    # train the network with m memory states
    for i in range(1, n):
      for j in range(1, n):
        weight = 0
        for zeta in range(0, m):
            weight = weight + (MemoryMatrix[zeta][i] * MemoryMatrix[zeta][j] * c[i][j])
        W[i][j] = (weight / n)
        W[j][i] = W[i][j]

    # evolve according to hopfield dynamics, n_i iterations
    for z in range(0, n_i):
        i = random.randint(1, n-1)
        for j in range (1, n):
         s = sum([W[i][j] * u[j] * c[i][j]])
        u[i] = 1 if s > 0 else -1 if s < 0 else u[i]
        z = z + 1

    # calculate hamming distances
    overlaptemp = 0
    for i in range(1, n):
        overlaptemp += u[i] * MemoryMatrix[1][i]

    X.append(alpha)
    Y.append(overlaptemp)

col_totalsEavg = [(x / n) for x in Y]

# set up figure and axes
plt.plot(X, col_totalsEavg, color="red")
plt.ylabel("% of errors")
plt.xlabel("alpha")
png = ".png"
suffix = datetime.datetime.now().strftime("%m%d_%H%M%S")
filename = "_".join([str(n), suffix])
plt.autoscale()
plt.savefig(filename, dpi=200, bbox_inches = "tight")
plt.show()

np.savetxt(filename + "-s-Q.csv", np.column_stack((X, col_totalsEavg)), delimiter=",", fmt='%s')