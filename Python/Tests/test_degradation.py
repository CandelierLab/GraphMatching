import os

import project
from Graph import *
from  Comparison import *

import paprint as pa

os.system('clear')

# === Parameters ===========================================================

directed = False

nA = 5
p = 0.5

algo = 'FAQ'

type = 'ed_rm'
delta = 0.5
# localization = 'first'
localization = False

# --------------------------------------------------------------------------

np.random.seed(0)

# ==========================================================================

# print('p', p)

# --- Random graphs

Ga = Gnp(nA, p, directed=directed)
Ga.add_edge_attr('rand')

# Ga.print()

Gb, gt = Ga.degrade(type, delta, localization=localization)

Ga.print()
Gb.print()
pa.matrix(Ga.Adj, highlight=Ga.Adj!=Gb.Adj)

print(gt.__dict__)

# C = Comparison(Ga, Gb)
# M = C.get_matching(algorithm=algo)
# M.compute_accuracy()

# print(M)
