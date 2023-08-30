import os

import project
from Network import *
from  Comparison import *

os.system('clear')

# --- Parameters -----------------------------------------------------------

nA = 5
p = 0.1

rho = 0.5

# --------------------------------------------------------------------------

Net = Network(nA)
Net.set_rand_edges('ER', p)
Net.add_edge_attr('rand', name='test')
Net.add_node_attr('rand', name='test2')

Net.print()

Sub, Icor = Net.subnet(round(nA*rho))

Sub.print()

# M = matching(Net, Sub, nIter=10, verbose=True)

# print(M)

# # Correct matches
# print(np.count_nonzero([Icor[m[1]]==m[0] for m in M])/Sub.nNd)