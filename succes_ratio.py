import os
import numpy as np
import matplotlib.pyplot as plt

import Network
from  Comparison import *

os.system('clear')

# === Parameters ===========================================================

Nmax = 15
Nsub = list(range(1, Nmax+1))

p = 0.5

nIter = 100

# ==========================================================================

rho = []

for n in Nsub:

  print(n)

  r = np.empty(nIter)

  for i in range(nIter):

    Net = Network.Random(Nmax, p, method='ER')
    Sub, Idx = Net.subnet(n)

    if not Sub.Adj.any():
      r[i] = 0

    else:

      M = matching(Net, Sub)

      # Correct matches
      r[i] = np.count_nonzero([Idx[m[1]]==m[0] for m in M])/n

  rho.append(np.mean(r))

# === Display ==============================================================

plt.style.use('dark_background')

fig, ax = plt.subplots()

ax.axhline(y = 1/Nmax, color = 'w', linestyle = ':')

ax.plot(Nsub, rho)

ax.set_xlabel('Number of subgraph nodes')
ax.set_xlim(1, Nmax)
ax.set_ylabel('Correct matches ratio')
ax.set_ylim(0, 1)

plt.show()