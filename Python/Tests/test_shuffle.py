import os

import project
from Network import *
from  Comparison import *

os.system('clear')

# --- Parameters -----------------------------------------------------------

nA = 5
p = 0.5

algo = 'GASM'

np.random.seed(0)

# --------------------------------------------------------------------------

# --- Random graphs

# Ga = Network(nA)
# Ga.set_rand_edges('ER', p_edges=p)
# # Ga.add_edge_attr('rand', name='test_edge')
# # Ga.add_node_attr('rand', name='test_node')

#  --- Balanced trees

Ga = Network(nx=nx.balanced_tree(2, 3))

Gb, Idx = Ga.shuffle()

C = Comparison(Ga, Gb, algorithm=algo)
M = C.get_matching()
M.compute_accuracy(Idx)

print(M)

# print(f'{len(M)} Matchings:')

# for m in M:
#   print(m)
