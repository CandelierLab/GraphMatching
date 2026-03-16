import os

import project
from Graph import *
from  Comparison import *

import paprint as pa

os.system('clear')

# === Parameters ===========================================================

directed = True

nA = 10
p = 2/nA

algo = 'GASM'

type = 'vx_rm'
delta = 0.1
# localization = 'first'
localization = False

# --------------------------------------------------------------------------

np.random.seed(0)

# ==========================================================================

# print('p', p)

res = []
nIter = 1

for iter in range(nIter):

  # --- Random graphs

  Ga = Gnp(nA, p, directed=directed)
  Ga.add_edge_attr('gauss')

  Gb, gt = Ga.degrade(type, delta, shuffle=False)

  # Ga.print()
  # Gb.print()
  
  C = Comparison(Ga, Gb)
  M = C.get_matching(algorithm=algo)
  M.compute_accuracy(gt)

  print(M)
  # print(gt)

  res.append(M.accuracy)

print(f'{nIter} iterations, <gamma> = {np.mean(res)}')