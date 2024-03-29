import os

import project
from Graph import *
from  Comparison import *
import paprint as pa

os.system('clear')

# --- Parameters -----------------------------------------------------------

nA = 10

# ER edge ratio
p = 0.01

# Subgraph ratio
rho = 1

# --------------------------------------------------------------------------

Net = Network(nA)
Net.set_rand_edges('ER', n_epn=0.5)
# Net.add_edge_attr('rand', name='test')
# Net.add_node_attr('rand', name='node_attr_1')

# Net.print()

Sub, Icor = Net.subnet(round(nA*rho))

# Sub.print()

# print(Icor)

pa.line(f'mA = {Net.nEd}')

M = matching(Net, Sub)

M.compute_accuracy(Icor)
print(M)

# Correct matches
# print(np.count_nonzero([Icor[m[1]]==m[0] for m in M])/Sub.nNd)