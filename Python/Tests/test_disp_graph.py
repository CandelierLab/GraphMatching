import os

import project
from Network import *
from  Comparison import *

# import AE.Network.ANN

os.system('clear')

# --- Parameters -----------------------------------------------------------

nA = 20

# --------------------------------------------------------------------------

'''
TO DO
  - Debug this
  - Add Gnp
  - Perform degradation
  - Rewrite and rerun all Analysis.Computation
'''

# Net = Network(nA)
# Net.set_rand_edges('ER', p_edges=0.1)

Net = Gnm(10,10)

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