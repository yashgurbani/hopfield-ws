import matplotlib.pyplot as plt
import networkx as nx

n = 32

fig = plt.figure(figsize=(40, 40))

g = nx.complete_graph(n)
nx.write_gexf(g, "test.gexf")