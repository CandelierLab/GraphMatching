import os

import project
from Network import *
from  Comparison import *

os.system('clear')

# --- Parameters -----------------------------------------------------------

nA = 5
p = 0

rho = 0.5

# --------------------------------------------------------------------------

Net = Network(nA)
Net.set_rand_edges('ER', p)
Net.add_edge_attr('rand', name='test_edge')
Net.add_node_attr('rand', name='test_node')

Net.print()

Met, Icor = Net.shuffle()

print('Correspondence: ', Icor)
Met.print()

M = matching(Net, Met, verbose=True)

print(M)

# Correct matches
print(np.count_nonzero([Icor[m[1]]==m[0] for m in M])/nA)