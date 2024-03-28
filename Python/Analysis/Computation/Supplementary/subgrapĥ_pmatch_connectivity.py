import os
import numpy as np
import time
import matplotlib.pyplot as plt

import project
from Graph import *
from Comparison import *

os.system('clear')

# === Parameters ===========================================================

nA = 20
nB = 15

p = 0.2

power = [1, 0.5]

nRun = 100000
# nRun = 100

# ==========================================================================

R = []
T = []

for pow in power:

  r = np.zeros(2*nA)
  t = np.zeros(2*nA)

  print('{:d} iterations with subgraph of size {:d} ...'.format(nRun, nB), end='')
  start = time.time()

  g = np.empty(nRun)

  for i in range(nRun):

    Net = Network(nA)
    Net.set_rand_edges('ER', p)

    Sub, Idx = Net.subnet(nB)

    Conn = np.sum(Net.Adj[Idx,:], axis=1) + np.sum(Net.Adj[:,Idx], axis=0)

    M = matching(Net, Sub, nIter=20, power=pow)

    # Correct matches
    CM = [Idx[m[1]]==m[0] for m in M]

    for k, c in enumerate(CM):
      r[Conn[k]] += int(c)
      t[Conn[k]] += 1

  R.append(r)
  T.append(t)

  print('{:.02f} sec'.format((time.time() - start)))

# === Display =================================================================

plt.style.use('dark_background')

fig, ax = plt.subplots()

for i in range(len(power)):
  ax.axhline(np.sum(R[i])/np.sum(T[i]), linestyle='--', color='w')
  ax.plot(R[i]/T[i], '.-', label=power[i])

ax.set_xlabel('Connectivity')
ax.set_ylabel('Probability to be correctly matched')

ax.set_ylim(0,1)
ax.legend()

plt.show()