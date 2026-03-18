import numpy as np
import networkx  as nx
import matplotlib.pyplot as plt




# G = nx.random_internet_as_graph(100, seed=None)

# G = nx.davis_southern_women_graph()

# G = nx.connected_caveman_graph(10, 10)
G = nx.relaxed_caveman_graph(5, 10, 0.1)

nx.draw(G, node_size=50)

plt.show()
