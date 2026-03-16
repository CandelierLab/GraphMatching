import os

import project
from Graph import *
from  Comparison import *

# import AE.Network.ANN

os.system('clear')

# --- Parameters -----------------------------------------------------------

nA = 10

# --------------------------------------------------------------------------

# G = Gnp(10, 0.1, directed=True)
G = star_branched(3, 5, directed=False)

# G = Network(nx=nx.balanced_tree(2, 3, create_using=nx.DiGraph))

G.print()

G.display()

# --- Display with AE ------------------------------------------------------

# N = AE.Network.ANN.ANN()
# print(N)

# for i in range(nA):
#   N.add_node()

# for e in Net.edges:
#   N.add_edge(e[0], e[1])

# N.nodeRadius = 0.01
# N.nodeFontSize = 6
# N.edgeFontSize = 1

# N.show()