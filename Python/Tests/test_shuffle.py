import os

import project
from Graph import *
from  Comparison import *

import paprint as pa

os.system('clear')

# === Parameters ===========================================================

nA = 10
# p = 0.9

algo = 'GASM'

# np.random.seed(0)

# --------------------------------------------------------------------------

p = np.log(nA)/nA

# ==========================================================================

# --- Random graphs

Ga = Gnp(nA, p, directed=False)
# # Ga.add_edge_attr('rand', name='test_edge')
# # Ga.add_node_attr('rand', name='test_node')

Gb, Idx = Ga.shuffle()

Ga.print()
# print(Idx)
# Gb.print()

C = Comparison(Ga, Gb, verbose=True)
M = C.get_matching(algorithm=algo)
M.compute_accuracy(Idx)

print(M)
# pa.matrix(C.X)

