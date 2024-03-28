import os

import project
from Graph import *
from  Comparison import *

# import AE.Network.ANN

os.system('clear')

# --- Parameters -----------------------------------------------------------

nA = 10

# --------------------------------------------------------------------------

# Net = Gnp(10, 0.1, directed=True)
Net = star_branched(4, 5, directed=True)

# Net = Network(nx=nx.balanced_tree(2, 3, create_using=nx.DiGraph))

Net.print()

Net.display()

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