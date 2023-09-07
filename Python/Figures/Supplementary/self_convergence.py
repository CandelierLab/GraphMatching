import os
import numpy as np
import time
import matplotlib.pyplot as plt

import project
from Network import *
from Comparison import *

os.system('clear')

# === Parameters ===========================================================

n = 1000

# Average number of edges per node
l_nepn = [0.25, 0.5, 0.75, 1, 2, 5]

l_nIter = range(11)

# ==========================================================================

l_rho = []

for nepn in l_nepn:

  print('Nepn {:.01f} ...'.format(nepn), end='')
  start = time.time()

  # --- Network

  Net = Network(n)
  Net.set_rand_edges('ER', int(nepn*n))

  Set, Icorr = Net.shuffle()

  # --- Convergence

  rho = []

  for nIter in l_nIter:

    M = matching(Net, Set, nIter=nIter)

    # Correct matches
    rho.append(np.count_nonzero([Icorr[m[1]]==m[0] for m in M])/n)

  l_rho.append(rho)

  print('{:.02f} sec'.format((time.time() - start)))

# === Display =================================================================

# plt.style.use('dark_background')

fig, ax = plt.subplots()

for i, rho in enumerate(l_rho):

  ax.plot(l_nIter, rho, '.-', label=l_nepn[i])

ax.set_xlabel('Iteration')
ax.set_ylabel(r'Ratio of correct matches $\rho$')
ax.set_title('legend: average number of edge per node')

# ax.set_xlim(0, max(l_nIter))
ax.set_ylim(0,1)
ax.legend()

plt.show()