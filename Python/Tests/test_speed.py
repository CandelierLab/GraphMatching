import os

from Graph import *
from  Comparison import *

os.system('clear')

# === Parameters ===========================================================

directed = True

nA = 100
p = 0.5

nRun = 10

algo = 'GASM'

# ==========================================================================

T = []

for r in range(nRun+1):

  Ga = Gnp(nA, p, directed=directed)
  Gb, gt = Ga.shuffle()

  C = Comparison(Ga, Gb, verbose=False)

  # Matching
  M = C.get_matching(algorithm=algo, GPU=True)

  # Remove first execution
  if r==0: continue

  # Store times
  T.append(M.time['total'])

print('Average time:', np.mean(T), 'ms')

