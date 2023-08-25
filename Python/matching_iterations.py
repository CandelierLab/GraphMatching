import os
import numpy as np
import matplotlib.pyplot as plt

import Network
from  Comparison import *

os.system('clear')

# === Parameters ===========================================================

Nmax = 50

p = 0.5

nIterMax = 11

# ==========================================================================

Net = Network.Random(Nmax, p, method='ER')

ni = []
rho = []

for nIter in range(1,nIterMax):
  
  ni.append(nIter)

  M = matching(Net, Net, weight_constraint=False, nIter=nIter)

  # Correct matches
  rho.append(np.count_nonzero([m[1]==m[0] for m in M])/Nmax)

print(rho)

# === Display ==============================================================

plt.style.use('dark_background')

fig, ax = plt.subplots()

ax.axhline(y = 1/Nmax, color = 'w', linestyle = ':')

ax.plot(ni, rho)

ax.set_xlabel('Number of iterations')
ax.set_xlim(1, nIterMax)
ax.set_ylabel('Correct matches ratio')
ax.set_ylim(0, 1.1)

plt.show()