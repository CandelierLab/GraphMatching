import os

import project
from Graph import *
from  Comparison import *

import paprint as pa

os.system('clear')

# === Parameters ===========================================================

nA = 15
# p = np.log(nA)/nA
p = 0.2

algo = 'Zager'

delta = 0.5
# localization = 'last'
localization = False

# --------------------------------------------------------------------------

# np.random.seed(0)

# ==========================================================================

# print('p', p)

# --- Random graphs

Ga = Gnp(nA, p, directed=True)
# Ga.add_vrtx_attr('rand')
# Ga.add_edge_attr('rand')

Gb, gt = Ga.subgraph(delta=delta, localization=localization)

Ga.print()
print(gt.__dict__)
Gb.print()

C = Comparison(Ga, Gb, verbose=True)
M = C.get_matching(algorithm=algo)

# M = Matching(Ga, Gb)
# M.from_lists(gt.Ia, gt.Ib)

M.compute_accuracy(gt)

print(M)