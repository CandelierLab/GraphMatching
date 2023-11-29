import os
import numpy as np
import time
import matplotlib.pyplot as plt

import project
from Network import *
from Comparison import *

os.system('clear')

# === Parameters ===========================================================

n = 1

# Average number of edges per node
l_nepn = [1]

l_nIter = range(1,10)

# ==========================================================================

l_t = []

for nepn in l_nepn:

  print('Nepn {:.01f} ...'.format(nepn), end='')
  start = time.time()
  
  # --- Network

  Net = Network(n)
  Net.set_rand_edges('ER', int(nepn*n))

  Set, Icorr = Net.shuffle()

  # --- Convergence

  t = []

  for nIter in l_nIter:

    tref = time.perf_counter_ns()
    M = matching(Net, Set, nIter=nIter)
    t.append((time.perf_counter_ns() - tref)*1e-9)

  l_t.append(t)

  print('{:.02f} sec'.format((time.time() - start)))

# === Display =================================================================

# plt.style.use('dark_background')

fig, ax = plt.subplots()

for i, t in enumerate(l_t):

  ax.plot(l_nIter, t, '.-', label=l_nepn[i])

  # Fit
  x = np.arange(2,10)
  f = np.polyfit(x, t[1:], 1)

  ax.plot(x, f[1]+f[0]*x, 'r-')


ax.set_xlabel('Iteration')
ax.set_ylabel('Iteration time')
ax.set_title('legend: average number of edge per node')

# ax.set_xlim(0, max(l_nIter))
# ax.set_ylim(0,1)
ax.legend()

plt.show()