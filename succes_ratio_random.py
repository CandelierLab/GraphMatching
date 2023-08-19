import os
import numpy as np
import matplotlib.pyplot as plt

import Network
from  Comparison import *

os.system('clear')

# === Parameters ===========================================================

Nmax = 50
Nsub = list(range(5, Nmax+1, 5))

p = 0.5

nIter = 10

# ==========================================================================

rho_bin = []
rho_rand = []

for n in Nsub:

  print(n)

  rb = np.empty(nIter)
  rr = np.empty(nIter)

  for i in range(nIter):

    # --- Random weights

    Net = Network.Random(Nmax, p, method='rand')
    Sub, Idx = Net.subnet(n)

    if not Sub.Adj.any():
      rr[i] = 0

    else:

      M = matching(Net, Sub)

      # Correct matches
      rr[i] = np.count_nonzero([Idx[m[1]]==m[0] for m in M])/n

    # --- Binarization

    th = int(np.round(p*Net.nNd**2))
    Net.Adj = (Net.Adj < np.sort(Net.Adj.flatten())[th])
    Sub = Net.subnet(Idx)

    # Reset edges
    Net.adj2edges(force=True)
    Sub.adj2edges(force=True)

    if not Sub.Adj.any():
      rb[i] = 0

    else:

      M = matching(Net, Sub)

      # Correct matches
      rb[i] = np.count_nonzero([Idx[m[1]]==m[0] for m in M])/n

  rho_bin.append(np.mean(rb))
  rho_rand.append(np.mean(rr))

# === Display ==============================================================

plt.style.use('dark_background')

fig, ax = plt.subplots()

ax.axhline(y = 1/Nmax, color = 'w', linestyle = ':')

ax.plot(Nsub, rho_rand, label='rand')
ax.plot(Nsub, rho_bin, label='bin')

ax.legend()

ax.set_xlabel('Number of subgraph nodes')
ax.set_xlim(1, Nmax)
ax.set_ylabel('Correct matches ratio')
ax.set_ylim(0, 1)

plt.show()