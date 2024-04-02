import os

import project
from Graph import *
from  Comparison import *

import paprint as pa

os.system('clear')

# === Parameters ===========================================================

nA = 5
# p = np.log(nA)/nA
p = 0.5

algo = 'FAQ'

delta = 0.5
localization = 'last'
# localization = False

# --------------------------------------------------------------------------

np.random.seed(0)

# ==========================================================================

# print('p', p)

# --- Random graphs

Ga = Gnp(nA, p, directed=False)
Ga.add_vrtx_attr('rand')
Ga.add_edge_attr('rand')


Gb, Idx = Ga.subgraph(delta=0.5, localization=localization)

Ga.print()
# print(Idx)
Gb.print()

# pa.matrix(Ga.Adj, highlight=Ga.Adj!=Gb.Adj)

# C = Comparison(Ga, Gb)
# M = C.get_matching(algorithm=algo)
# M.compute_accuracy()

# print(M)
