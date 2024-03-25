import os
import time
import matplotlib.pyplot as plt

import project
from Network import *
from Comparison import *

os.system('clear')

# === Parameters ===========================================================

n = 1000

# Average number of edges per node
l_nepn = [0.25, 0.5, 0.75, 1, 1.5, 2, 3]

nIterMax = 6

# ==========================================================================

accuracy = []

for nepn in l_nepn:

  print('Nepn {:.01f} ...'.format(nepn), end='')
  start = time.time()

  # --- Network

  NetA = Network(n)
  NetA.set_rand_edges('ER', n_epn=nepn)

  NetB, Idx = NetA.shuffle()

  # --- Convergence

  # Scores
  C = Comparison(NetA, NetB)

  l_gamma = []

  for nIter in range(nIterMax):

    M = C.get_matching(algorithm='GASM', nIter=nIter, force=True)
    M.compute_accuracy(Idx)   
    l_gamma.append(M.accuracy)

  accuracy.append(l_gamma)

  print('{:.02f} sec'.format((time.time() - start)))

# === Display =================================================================

# plt.style.use('dark_background')
fig, ax = plt.subplots()

for i, gamma in enumerate(accuracy):

  ax.plot(gamma, '.-', label=l_nepn[i])

ax.set_xlabel('Iteration')
ax.set_ylabel(r'$\gamma$')
ax.set_title('legend: average number of edge per node')

# ax.set_xlim(0, max(l_nIter))
ax.xaxis.grid(True)
ax.set_ylim(0,1)
ax.legend()

plt.show()