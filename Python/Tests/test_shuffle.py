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

Gb, gt = Ga.shuffle()

Ga.print()
print(gt.__dict__)
Gb.print()

# C = Comparison(Ga, Gb, verbose=True)
# M = C.get_matching(algorithm=algo)

M = Matching(Ga, Gb)
M.from_lists(gt.Ia, gt.Ib)

M.compute_accuracy(gt)

print(M)
# pa.matrix(C.X)

