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

# --------------------------------------------------------------------------

ds = 'directed' if directed else 'undirected'

fname = project.root + f'/Files/Degradation/ER/{ds}_ksi=1_nA={nA:d}_nRun={nRun:d}.csv'

# ==========================================================================

if os.path.exists(fname):

  # Load data
  df = pd.read_csv(fname)

  # Retrieve l_delta and l_precision

  l_delta = np.unique(df.delta)
  l_precision = np.unique(df.precision)
  l_precision = l_precision[~np.isnan(l_precision)]

print(l_precision)
# --- Display --------------------------------------------------------------

plt.style.use('dark_background')
fig, ax = plt.subplots(1,2, figsize=(12,6))

# --- FAQ

data = df.loc[df['algo']=='FAQ']
  
ax[0].plot(l_delta, data.g, '--', label=f'FAQ')
ax[1].plot(l_delta, data.q, '--', label=f'FAQ')

# --- GASM

data = df.loc[(df['algo']=='GASM') & pd.isnull(df['precision'])]

ax[0].plot(l_delta, data.g, '--', label=f'GASM')
ax[1].plot(l_delta, data.q, '--', label=f'GASM')

# --- GASM with precision

for p in l_precision:

  data = df.loc[(df['algo']=='GASM') & (df['precision']==p)]
  
  ax[0].plot(l_delta, data.g, '-', label=f'GASM $\\rho={p:.03f}$')
  ax[1].plot(l_delta, data.q, '-', label=f'GASM $\\rho={p:.03f}$')

# ax[0].set_xscale('log')
ax[0].set_yscale('log')

# ax[0].set_xlim([0, 1])
# ax[0].set_ylim([0.005, 1])

# ax[1].set_xlim([0, 1])
# ax[1].set_ylim([0.9, 1])

ax[0].set_xlabel('$\delta$')
ax[1].set_xlabel('$\delta$')

ax[0].set_ylabel('$\gamma$')
ax[1].set_ylabel('$q$')

ax[0].legend()
ax[1].legend()

plt.show()