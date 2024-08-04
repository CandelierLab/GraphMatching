import os

import project
from Graph import *
from  Comparison import *

import paprint as pa

os.system('clear')

# === Parameters ===========================================================

directed = True

nA = 100
p = 0.5

nRun = 10

algo = 'GASM'

# np.random.seed(0)

# --------------------------------------------------------------------------

p = np.log(nA)/nA

# ==========================================================================

T = []

for r in range(nRun+1):

  Ga = Gnp(nA, p, directed=directed)
  Gb, gt = Ga.shuffle()

  C = Comparison(Ga, Gb, verbose=False)

  # First execution
  M = C.get_matching(algorithm=algo, GPU=True)

  if r==0: continue

  # Subsequent executions
  M = C.get_matching(algorithm=algo, GPU=True)
  T.append(M.time['total'])

print('Average time:', np.mean(T), 'ms')

