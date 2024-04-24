'''
Balanced tree: average gamma and q
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
r = 2

# --- Plot parameters

err_alpha = 0.2
lw = 3
fontsize = 24

# Colors
c = {'2opt': '#CC4F1B', 'e2opt': '#FF9848',
     'FAQ': '#FFA500', 'eFAQ': '#FACC2E',
     'Zager': '#1B2ACC', 'eZager': '#089FFF',
     'GASM': '#3F7F4C', 'eGASM':'#7EFF99'}

# ==========================================================================

plt.rcParams.update({'font.size': fontsize})
fig, ax = plt.subplots(1, 2, figsize=(20,10))

l_h = None

for algo in l_algo:

  # --- Load data ----------------------------------------------------------

  datapath = project.root + f'/Files/Self-matching/BT/{algo}_r={r:d}.csv'

  if os.path.exists(datapath):

    # Load data
    data = pd.read_csv(datapath)

    # Retrieve l_h
    l_h = np.unique(data.h)

    # --- Plots ------------------------------------------------------------

    # Accuracy
    ax[0].plot(data.h, data.g, '-', color=c[algo], linewidth=lw, label=algo)
    
    # Structural quality
    ax[1].plot(data.h, data.q, '-', color=c[algo], linewidth=lw, label=algo)
    ax[1].fill_between(data.h, data.q - data.q_std, data.q + data.q_std, alpha=err_alpha, facecolor=c['e'+algo])


ax[0].plot(l_h, (l_h+1)/(2**(l_h+1) - 1), '--', color='k', linewidth=lw, label='Theoretical')

# --- Figure options -------------------------------------------------------

ax[0].set_yscale('log')

ax[0].set_xticks(range(2,11,2))
ax[1].set_xticks(range(2,11,2))

ax[0].set_ylim([0, 1])
ax[1].set_ylim([0.75, 1])

ax[0].set_xlabel('$h$')
ax[1].set_xlabel('$h$')

ax[0].set_ylabel('$\gamma$')
ax[1].set_ylabel('$q_s$')

ax[0].legend()
ax[1].legend()

plt.show()