

import networkx as nx 
import matplotlib .pyplot as plt
import random
import itertools

G = nx.Graph()
n = 10
G.add_nodes_from([i for i in range(1, n + 1)])

for i in G.nodes():
    for j in G.nodes():
        if i != j:
            G.add_edge(i, j)

pos = nx.circular_layout(G)

nx.draw(G, pos, node_size = 2000, with_labels = 1)
plt.show()
nodes = G.nodes()
print(nodes)
tris_list = [list(x) for x in itertools.combinations(nodes, 3)]
print(tris_list)