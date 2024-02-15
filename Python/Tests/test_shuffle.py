import os

import project
from Network import *
from  Comparison import *

os.system('clear')

# --- Parameters -----------------------------------------------------------

nA = 5
p = 0.5

algo = 'GASP'

np.random.seed(0)

# --------------------------------------------------------------------------

Ga = Network(nA)
Ga.set_rand_edges('ER', p_edges=p)
# Ga.add_edge_attr('rand', name='test_edge')
# Ga.add_node_attr('rand', name='test_node')

Ga.print()

Gb, Icor = Ga.shuffle()

# print('Correspondence: ', Icor)
# Gb.print()

M = matching(Ga, Gb, verbose=True)

print(M)

# print(f'{len(M)} Matchings:')

# for m in M:
#   print(m)
