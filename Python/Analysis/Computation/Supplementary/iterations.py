import os
import numpy as np
import time
from alive_progress import alive_bar
import matplotlib.pyplot as plt

import project
from Graph import *
from Comparison import *

os.system('clear')

# === Parameters ===========================================================

directed = True
nA = 100
nRun = 10

# Average degree
l_deg = [1] #[0.25, 0.5, 0.75, 1, 1.5, 2, 3]

# l_nIter = range(6)
l_nIter = range(-1,5)

# ==========================================================================

m_gamma = []
s_gamma = []

for deg in l_deg:

  l_g = []
  
  with alive_bar(nRun) as bar:

    bar.title(f'deg={deg}')

    for r in range(nRun):

      # --- Network

      Ga = Gnp(nA, deg/nA, directed=directed)
      Gb, gt = Ga.shuffle()

      # --- Convergence

      g = []

      for nIter in l_nIter:

        C = Comparison(Ga, Gb)
        M = C.get_matching(algorithm='GASM', nIter=nIter)
        M.compute_accuracy(gt)

        g.append(M.accuracy)
      
      l_g.append(g)

      bar()

  m_gamma.append(np.mean(l_g, axis=0))
  s_gamma.append(np.std(l_g, axis=0))

# === Display =================================================================

# plt.style.use('dark_background')

fig, ax = plt.subplots(figsize=(8,5))

for i, deg in enumerate(l_deg):

  ax.plot(l_nIter, m_gamma[i], '.-', label=deg)

ax.set_xlabel('iteration')
ax.set_ylabel(r'accuracy $\gamma$')
ax.legend()

# ax.set_xlim(0, max(l_nIter))
# ax.set_ylim(0,1)

plt.show()