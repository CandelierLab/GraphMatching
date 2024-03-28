import os

import project
from Graph import *
from  Comparison import *

import paprint as pa

os.system('clear')

# === Parameters ===========================================================

nA = 5
# p = 0.9

algo = 'GASM'

# np.random.seed(0)

# --------------------------------------------------------------------------

p = np.log(nA)/nA

# ==========================================================================

# --- Random graphs

Ga = Gnp(nA, p)

# # Ga.add_edge_attr('rand', name='test_edge')
# # Ga.add_node_attr('rand', name='test_node')

# Ga = Network(nx=nx.circular_ladder_graph(5))

# Ga = star_branched(9,10)

Gb, Idx = Ga.shuffle()

Ga.print()
print(Idx)
Gb.print()

# C = Comparison(Ga, Gb, verbose=True)
# M = C.get_matching(algorithm=algo)
# M.compute_accuracy(Idx)

# print(M)
# pa.matrix(C.X)

