#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: yash
"""
import datetime
import matplotlib.pyplot as plt
import networkx as nx

Ldata = []
Cdata = []
Xax = []
Compdata = []

p = 0.001
while (p <= 1): #for this given random matrix, loop over different p values of small world graph
                 
        g = nx.watts_strogatz_graph(1000, 150, p)
        for X in (g.subgraph(x).copy() for x in nx.connected_components(g)):
         Compdata.append(nx.average_shortest_path_length(X))
         
        L0 = sum(Compdata)/len(Compdata)
        C0 = nx.average_clustering(g)
        print (L0, C0)
        Ldata.append(L0)
        Cdata.append(C0)
        Xax.append(p)
        p = p + 0.01

# set up figure and axes
f, ax = plt.subplots(1,1)

#plt.title('Variation of Network Performance with Rewiring Probability \n Iterations = %s |  k = %s | Ensemble Count= %s \n \n' %(n, k, ensembleCount))
# set up figure and axes
plt.subplot(2, 1, 1)
plt.plot(Xax, Cdata, color="blue")
plt.ylabel('clustering coefficient')
plt.ylabel("average quality q")

plt.subplot(2, 1, 2)
plt.ylabel('average path length')
plt.xlabel('rewiring probability p')
plt.plot(Xax, Ldata, color="red")

suffix = datetime.datetime.now().strftime("%m%d_%H%M%S")
plt.autoscale()
plt.savefig(suffix, dpi=200, bbox_inches = "tight")
plt.show()


