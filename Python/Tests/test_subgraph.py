import os

import project
from Network import *
from  Comparison import *
import paprint as pa

os.system('clear')

# --- Parameters -----------------------------------------------------------

nA = 20

# ER edge ratio
p = 0.01

# Subgraph ratio
rho = 0.5

# --------------------------------------------------------------------------

Net = Network(nA)
Net.set_rand_edges('ER', p)
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