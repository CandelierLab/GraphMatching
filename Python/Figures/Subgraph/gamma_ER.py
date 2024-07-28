'''
Degradation of ER: average gamma and q
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import project

os.system('clear')

# === Parameters ===========================================================

directed = True

nA = 200
nRun = 1000

l_cond = ['zeta=0', 'ksi=1', 'zeta=1']

# --- Plot parameters

err_alpha = 0.2
lw = 3
fontsize = 12

# Colors

# ==========================================================================

plt.rcParams.update({'font.size': fontsize})
fig, ax = plt.subplots(2, 3, figsize=(15,10))

cm =plt.cm.jet(np.linspace(0,1,7))
ax[0,1].set_prop_cycle(plt.cycler(color=cm))
ax[1,1].set_prop_cycle(plt.cycler(color=cm))
ax[0,2].set_prop_cycle(plt.cycler(color=cm))
ax[1,2].set_prop_cycle(plt.cycler(color=cm))

for i, directed in enumerate([False, True]):

  ds = 'directed' if directed else 'undirected' 

  for j, cond in enumerate(l_cond): 

    fname = project.root + f'/Files/Subgraph/ER/{ds}_{cond}_nA={nA}_nRun={nRun}.csv'

    if os.path.exists(fname):

      # Load data
      df = pd.read_csv(fname)

      # Retrieve l_delta
      l_delta = np.unique(df.delta)

    # --- FAQ

    data = df.loc[df['algo']=='FAQ']
      
    ax[i,j].plot(l_delta, data.g, '--', color='k', label=f'FAQ')

    # --- GASM

    match cond:
      case 'zeta=0':
        data = df.loc[df['algo']=='GASM']
      case _:
        data = df.loc[(df['algo']=='GASM') & pd.isnull(df['precision'])]

    ax[i,j].plot(l_delta, data.g, '.-', color='k', label=f'GASM')

    # --- GASM with precision

    if j:

      l_precision = np.unique(df.precision)
      l_precision = l_precision[~np.isnan(l_precision)]

      for p in l_precision:

        data = df.loc[(df['algo']=='GASM') & (df['precision']==p)]
        
        ax[i,j].plot(l_delta, data.g, '-', label=f'GASM $\\rho={p:.03f}$')

    # --- General plot settings

    ax[i,j].set_yscale('log')
    ax[i,j].set_ylim([0.002, 1.2])
    ax[i,j].set_ylabel('$\gamma$')

    ax[i,j].set_xlim([0, 1])
    ax[i,j].set_xlabel('$\delta$')

# --- Specific plot settings

ax[1,1].legend()

plt.show()