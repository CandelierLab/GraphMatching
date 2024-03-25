import os

import project
from Network import *
from  Comparison import *

# import AE.Network.ANN

os.system('clear')

# --- Parameters -----------------------------------------------------------

nA = 20

# --------------------------------------------------------------------------

# Net = Network(nA)
# Net.set_rand_edges('ER', p_edges=0.1)

Net = Network(nx=nx.balanced_tree(2, 3))

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


Net.display()

