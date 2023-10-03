import os

import project
from Network import *
from  Comparison import *

os.system('clear')

# --- Parameters -----------------------------------------------------------

nA = 10
p = 2/nA

rho = 0.5

# --------------------------------------------------------------------------

Net = Network(nA)
Net.set_rand_edges('ER', p)
# Net.add_edge_attr('rand', name='test')
Net.add_node_attr('rand', name='node_attr_1')

# Net.print()

Sub, Icor = Net.subnet(round(nA*rho))

# Purely structural scores
# Xs = scores(Net, Sub, nIter=nIter)[0]

# Structure & attribute scores
# X = scores(Net, Net)[0]
# print(X)

M = matching(Net, Sub)

print(M)

# Correct matches
print('gamma:', np.count_nonzero([Icor[m[1]]==m[0] for m in M])/Sub.nNd)