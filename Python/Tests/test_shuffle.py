import os

import project
from Graph import *
from  Comparison import *

import paprint as pa

os.system('clear')

# === Parameters ===========================================================

directed = False
nA = 5
p = 0.25

algo = 'GASM'

np.random.seed(0)

# --------------------------------------------------------------------------

# p = np.log(nA)/nA

# ==========================================================================

# --- Random graphs

Ga = Gnp(nA, p, directed=directed)
Ga.add_edge_attr('rand', name='test_edge')
# Ga.add_vrtx_attr('rand', name='test_node')

# print(Ga.nV**2, Ga.nEd, Ga.nEd/Ga.nV**2)

Ga.print()

Gb, gt = Ga.shuffle()

print(gt.__dict__)
Gb.print()


# C = Comparison(Ga, Gb, verbose=True)
# M = C.get_matching(algorithm=algo)
# M.compute_accuracy(gt)

# print(M)

