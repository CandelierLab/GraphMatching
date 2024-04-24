'''
Circular ladder graph: average gamma and q
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

# --- Plot parameters

err_alpha = 0.2
lw = 3
fontsize = 24
markersize = 10

# Colors
c = {'2opt': '#CC4F1B', 'e2opt': '#FF9848',
     'FAQ': '#FFA500', 'eFAQ': '#FACC2E',
     'Zager': '#1B2ACC', 'eZager': '#089FFF',
     'GASM': '#3F7F4C', 'eGASM':'#7EFF99'}

# ==========================================================================

plt.rcParams.update({'font.size': fontsize})
fig, ax = plt.subplots(1,2, figsize=(20,10))

for algo in l_algo:

  fname = project.root + f'/Files/Self-matching/CL/{algo}_CL.csv'

  if os.path.exists(fname):

    # Load data
    data = pd.read_csv(fname)

    # Retrieve l_h
    l_n = np.unique(data.n)

    # Accuracy
    ax[0].plot(data.n, data.g, '.', color=c[algo], linewidth=lw, markersize=markersize, label=algo)

    # Structural quality
    ax[1].plot(data.n, data.q, '-', color=c[algo], linewidth=lw, label=algo)
    ax[1].fill_between(data.n, data.q - data.q_std, data.q + data.q_std, alpha=err_alpha, facecolor=c['e'+algo])

ax[0].plot(l_n, 1/2/l_n, '--', color='k', linewidth=lw, label='Theoretical')

# --- Figure options -------------------------------------------------------

ax[0].set_xscale('log')
ax[0].set_yscale('log')

ax[0].set_ylim([0,1])
ax[1].set_ylim([0, 1])

ax[0].set_xlabel('$n$')
ax[1].set_xlabel('$n$')

ax[0].set_ylabel('$\gamma$')
ax[1].set_ylabel('$q_s$')

ax[0].legend()
ax[1].legend()

plt.show()