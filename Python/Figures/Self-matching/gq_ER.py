'''
Erdo-Renyi: average gamma and q
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import project

os.system('clear')

# === Parameters ===========================================================

l_algo = ['FAQ', '2opt', 'Zager', 'GASM']

directed = False
nA = 20

# --- Plot parameters

err_alpha = 0.2
lw = 3
fontsize = 48

# Colors
c = {'2opt': '#CC4F1B', 'e2opt': '#FF9848',
     'FAQ': '#FFA500', 'eFAQ': '#FACC2E',
     'Zager': '#1B2ACC', 'eZager': '#089FFF',
     'GASM': '#3F7F4C', 'eGASM':'#7EFF99'}

# --------------------------------------------------------------------------

ds = 'directed' if directed else 'undirected'

# ==========================================================================

plt.rcParams.update({'font.size': fontsize})
fig, ax = plt.subplots(1, 2, figsize=(25,10))

for algo in l_algo:

  datapath = project.root + f'/Files/Self-matching/ER/{algo}_{ds}_nA={nA:d}.csv'

  if os.path.exists(datapath):

    # Load data
    data = pd.read_csv(datapath)

    # Retrieve l_p

    l_p = np.unique(data.p)

    # Accuracy
    ax[0].plot(data.p, data.g, '-', color=c[algo], linewidth=lw, label=algo)

    # Structural quality
    ax[1].plot(data.p, data.q, '-', color=c[algo], linewidth=lw, label=algo)
    ax[1].fill_between(data.p, data.q - data.q_std, data.q + data.q_std, alpha=err_alpha, facecolor=c['e'+algo])

# --- Figure options -------------------------------------------------------

ax[0].set_ylim([0, 1])
ax[1].set_ylim([0.4, 1])

ax[0].set_xlabel('$p$')
ax[1].set_xlabel('$p$')

ax[0].set_ylabel('$\gamma$')
ax[1].set_ylabel('$q_s$')

ax[0].legend()
ax[1].legend()

plt.show()