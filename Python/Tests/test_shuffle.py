import os

import project
from Network import *
from  Comparison import *

os.system('clear')

# --- Parameters -----------------------------------------------------------

nA = 5
p = 0.1

rho = 0.5

algo = 'GASP'

# --------------------------------------------------------------------------

Net = Network(nA)
Net.set_rand_edges('ER', p)
Net.add_edge_attr('rand', name='test_edge')
Net.add_node_attr('rand', name='test_node')

Net.print()

Met, Icor = Net.shuffle()

# print('Correspondence: ', Icor)
# Met.print()

M = matching(Net, Met, algorithm=algo, all_solutions=True)

print(f'{len(M)} Matchings:')

for m in M:
  print(m)
