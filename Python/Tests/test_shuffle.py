import os

import project
from Graph import *
from  Comparison import *

import paprint as pa

os.system('clear')

# === Parameters ===========================================================

directed = True
nA = 5
p = 0.5

algo = 'GASM'
# algo = '2opt'

np.random.seed(0)

# --------------------------------------------------------------------------

# p = np.log(nA)/nA

# ==========================================================================

# Ga = star_branched(3, 3, directed=directed)

# --- Random graphs

Ga = Gnp(nA, 0, directed=directed)
# Ga.add_edge_attr('rand', name='test_edge')
# Ga.add_vrtx_attr('rand', name='test_node')

# print(Ga.nV**2, Ga.nEd, Ga.nEd/Ga.nV**2)

Gb, gt = Ga.shuffle()

Ga.print()
# Gb.print()

C = Comparison(Ga, Gb, verbose=True)

M = C.get_matching(algorithm=algo, GPU=False)
M.compute_accuracy(gt)
print(M)

M = C.get_matching(algorithm=algo, GPU=True, force=True)
M.compute_accuracy(gt)
print(M)