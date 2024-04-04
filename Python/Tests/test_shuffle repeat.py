import os
import time

import project
from Graph import *
from  Comparison import *

import paprint as pa

os.system('clear')

# === Parameters ===========================================================

nA = 20
p = 0.5

nRun = 100

# np.random.seed(3)

# --------------------------------------------------------------------------

# p = np.log(nA)/nA

# ==========================================================================

g_Zager = []
g_GASM = []

q_Zager = []
q_GASM = []

print('Computing ', end='', flush=True)
tref = time.time()

for run in range(nRun):

  # --- Random graphs

  Ga = Gnp(nA, p, directed=False)
  Gb, gt = Ga.shuffle()

  # --- Zager

  C = Comparison(Ga, Gb)
  M = C.get_matching(algorithm='Zager')
  M.compute_accuracy(gt)

  g_Zager.append(M.accuracy)
  q_Zager.append(M.structural_quality)

  #  --- GASM

  C = Comparison(Ga, Gb)
  M = C.get_matching(algorithm='GASM')
  M.compute_accuracy(gt)

  g_GASM.append(M.accuracy)
  q_GASM.append(M.structural_quality)

  if not run % 10:
    print('.', end='', flush=True)

print('{:.02f} sec'.format((time.time() - tref)))

print('Zager accuracy', np.mean(g_Zager))
print('GASM accuracy', np.mean(g_GASM))

print('Zager str qlty', np.mean(q_Zager))
print('GASM str qlty', np.mean(q_GASM))