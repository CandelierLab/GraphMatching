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
nRun = 100

# Average degree
l_deg = [0.25, 0.5, 0.75, 1, 1.5, 2, 3]

l_nIter = np.arange(-1,21)

# --- Plot parameters

err_alpha = 0.2
lw = 3
fontsize = 12

# ==========================================================================

m_gamma = []
s_gamma = []

k_tilde = []
g_tilde = []

for deg in l_deg:

  l_g = []
  k = []
  
  with alive_bar(nRun) as bar:

    bar.title(f'deg={deg}')

    for r in range(nRun):

      # --- Network

      Ga = Gnp(nA, deg/nA, directed=directed)
      Gb, gt = Ga.shuffle()

      k.append(Ga.diameter)

      # --- Convergence

      g = []

      for nIter in l_nIter:

        C = Comparison(Ga, Gb)
        M = C.get_matching(algorithm='GASM', nIter=nIter)
        M.compute_accuracy(gt)

        g.append(M.accuracy)
      
      l_g.append(g)

      bar()

  mg = np.mean(l_g, axis=0)
  m_gamma.append(mg)
  s_gamma.append(np.std(l_g, axis=0))

  k_tilde.append(np.mean(k))

  # --- Interpolation

  g_tilde.append(np.interp(k_tilde[-1], l_nIter, mg))

# === Display =================================================================

cm = plt.cm.jet(np.linspace(0,1,len(l_deg)))
fig, ax = plt.subplots(figsize=(8,5))

for i, deg in enumerate(l_deg):

  ax.plot(l_nIter, m_gamma[i], '.-', color=cm[i], label=deg)

  ax.fill_between(l_nIter, m_gamma[i] - s_gamma[i]/2, m_gamma[i] + s_gamma[i]/2, alpha=err_alpha, facecolor=cm[i])

  print(deg, ':', m_gamma[i][2])

ax.plot(k_tilde, g_tilde, 'k--')

ax.set_xlabel('iteration')
ax.set_ylabel(r'accuracy $\gamma$')
ax.legend()

ax.set_xlim(-1.05, max(l_nIter))
ax.set_ylim(0,1)

plt.show()