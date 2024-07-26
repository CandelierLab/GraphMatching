import os

import project
from Graph import *
from  Comparison import *

import paprint as pa

os.system('clear')

# === Parameters ===========================================================

# nA = 5
# p = 0.2

nA = 20
p = 0.5

algo = 'FAQ'

delta = 0.5
# localization = 'last'
# localization = False

# --------------------------------------------------------------------------

# np.random.seed(0)

# ==========================================================================

# print('p', p)

# --- Random graphs

Ga = Gnp(nA, p, directed=True)
# Ga.add_vrtx_attr('rand')
# Ga.add_edge_attr('rand')

Gb, gt = Ga.degrade('vx_rm', delta)

# Ga.print()
# print(gt.__dict__)
# Gb.print()

C = Comparison(Ga, Gb, verbose=True)
M = C.get_matching(algorithm=algo)
M.compute_accuracy(gt)

print(M)