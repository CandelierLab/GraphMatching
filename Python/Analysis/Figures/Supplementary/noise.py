import os, sys
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import project

os.system('clear')

# === Parameters ===========================================================

l_directed = [True, False]
# l_p = np.linspace(0, 1, 100)

nA = 20
nRun = 1000

l_eta = np.insert(np.logspace(-15, 0, 8), 0, 0)

dname = project.root + '/Files/Noise/'

# ==========================================================================

# --- Display

fig, ax = plt.subplots(1,2)
ax[0].set_prop_cycle(plt.cycler(color=plt.cm.turbo(np.linspace(0,1,len(l_eta)))))
ax[1].set_prop_cycle(plt.cycler(color=plt.cm.turbo(np.linspace(0,1,len(l_eta)))))
        

for directed in l_directed:

  sdir = 'directed' if directed else 'undirected'

  for eta in l_eta:

    # --- Load data

    fname = dname + f'ER_{sdir}_nA={nA}_eta={eta}_nRun={nRun}.csv'
    data = pd.read_csv(fname, index_col=0)

    l_p = np.unique(data.p)

    # --- Plot data

    g = data.groupby('p')['g'].mean()
    q = data.groupby('p')['q'].mean()

    if eta==0:
      if directed:
        ax[0].plot(l_p, g, '-', color='k', label=eta, zorder=1)
        ax[1].plot(l_p, q, '-', color='k', zorder=1)
      else:
        ax[0].plot(l_p, g, '--', color='k', zorder=1)
        ax[1].plot(l_p, q, '--', color='k', zorder=1)
    else:
      if directed:
        ax[0].plot(l_p, g, '-', label=eta, zorder=0)
        ax[1].plot(l_p, q, '-', zorder=0)
      else:
        ax[0].plot(l_p, g, '--', zorder=0)
        ax[1].plot(l_p, q, '--', zorder=0)


# --- Misc display

ax[0].set_xlabel(r'$p$')
ax[0].set_xlim(0, 1)
ax[0].set_ylabel(r'accuracy $\gamma$')
ax[0].set_ylim(0, 1)
# ax[0].set_yscale('log')

ax[1].set_xlabel(r'$p$')
ax[1].set_xlim(0, 1)
ax[1].set_ylabel(r'$q_s$')
ax[1].set_ylim(0, 1)
# ax[0].set_yscale('log')

ax[0].legend()

plt.show()