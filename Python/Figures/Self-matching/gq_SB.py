'''
Star-branched graph: average gamma and q
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import project
from Graph import *
from  Comparison import *

os.system('clear')

# === Parameters ===========================================================

l_algo = ['FAQ', '2opt', 'Zager', 'GASM']
directed = False

# l_k = np.array([2, 5, 10])
l_k = np.array([3])

# --- Plot parameters

err_alpha = 0.2
lw = 3
fontsize = 48

# Colors
c = {'2opt': '#CC4F1B', 'e2opt': '#FF9848',
     'FAQ': '#FFA500', 'eFAQ': '#FACC2E',
     'Zager': '#1B2ACC', 'eZager': '#089FFF',
     'GASM': '#3F7F4C', 'eGASM':'#7EFF99'}

ks = ['-', '--', ':']

# --------------------------------------------------------------------------

ds = 'directed' if directed else 'undirected'

# ==========================================================================

plt.rcParams.update({'font.size': fontsize})
fig, ax = plt.subplots(1, 2, figsize=(25,10))

l_h = None

for algo in l_algo:

  # --- Load data ----------------------------------------------------------

  datapath = project.root + f'/Files/Self-matching/SB/{algo}_{ds}_k=3.csv'

  if os.path.exists(datapath):

    # Load data
    df = pd.read_csv(datapath)

    # Retrieve l_k and l_n

    # l_k = np.unique(df.k).astype(int)
    l_n = np.unique(df.n).astype(int)

    print(algo, np.unique(df.nRun))

    # --- Plots

    for ki, k in enumerate(l_k):

      data = df.loc[df['k'] == k]

      # Accuracy    
      # ax[0].plot(data.n, data.g, linestyle=ks[ki], linewidth=lw, color=c[algo], label=f'{algo} $k = {k:d}$')
      ax[0].plot(data.n, data.g, '.', color=c[algo], label=f'{algo} $k = {k:d}$')

      # Structural quality
      if ki==0:
        ax[1].plot(data.n, data.q, linestyle=ks[ki], linewidth=lw, color=c[algo], label=f'{algo} $k = {k:d}$')
        ax[1].fill_between(data.n, data.q - data.q_std, data.q + data.q_std, alpha=err_alpha, facecolor=c['e'+algo])

ax[0].plot(data.n, (data.n+1)/(3*data.n+1), 'k--')

ax[0].set_xticks(range(1,11,3))
ax[1].set_xticks(range(1,11,3))

# ax[0].set_xscale('log')
ax[0].set_yscale('log')

ax[0].set_ylim([0, 1])
ax[1].set_ylim([0.4, 1])

ax[0].set_xlabel('$n$')
ax[1].set_xlabel('$n$')

ax[0].set_ylabel('$\gamma$')
ax[1].set_ylabel('$q_s$')

ax[0].legend()

plt.show()