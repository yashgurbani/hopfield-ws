#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: yash
"""
import datetime
import matplotlib.pyplot as plt
import networkx as nx
n = 128
k = 20
p = 0
while (p <= 1):

    filename = "_".join([str(n), str(k), str(p)])
    name = filename + ".gexf"
    g = nx.watts_strogatz_graph(128, 20, p)
    nx.write_gexf(g, name)
    p = p + 0.1



