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

nA = 20
p = np.log(nA)/nA

# nA = 30
# p = 0.25

nRun = 200

# --------------------------------------------------------------------------

ds = 'directed' if directed else 'undirected'

fname = project.root + f'/Files/Degradation/ER/{ds}_nA={nA:d}_p={p:.05f}_nRun={nRun:d}.csv'

print(fname)

# ==========================================================================

if os.path.exists(fname):

  # Load data
  df = pd.read_csv(fname)

  # Retrieve l_delta and l_meas

  l_delta = np.unique(df.delta)
  l_nvam = np.unique(df.nvam).astype(int)
  l_neam = np.unique(df.neam).astype(int)

# --- Display --------------------------------------------------------------

plt.style.use('dark_background')
fig, ax = plt.subplots(1,2, figsize=(12,6))

# --- FAQ

for neam in l_neam:

  data = df.loc[np.logical_and(df['algo']=='FAQ',
                               df['neam']==neam)]
  
  ax[0].plot(l_delta, data.g, '--', label=f'FAQ, $\\zeta_m=0$, $\\xi_m={neam}$')
  ax[1].plot(l_delta, data.q, '--', label=f'FAQ, $\\zeta_m=0$, $\\xi_m={neam}$')

# --- GASM

for nvam in l_nvam:
  for neam in l_neam:

    # Just one attribute at a time
    if nvam and neam: continue

    data = df.loc[np.logical_and(np.logical_and(df['algo']=='GASM', df['nvam']==nvam), df['neam']==neam)]
    
    ax[0].plot(l_delta, data.g, '-', label=f'GASM, $\\zeta_m={nvam}$, $\\xi_m={neam}$')
    ax[1].plot(l_delta, data.q, '-', label=f'GASM, $\\zeta_m={nvam}$, $\\xi_m={neam}$')

# ax[0].set_xscale('log')
ax[0].set_yscale('log')

ax[0].set_xlim([0, 1])
ax[0].set_ylim([0.005, 1])

ax[1].set_xlim([0, 1])
ax[1].set_ylim([0.9, 1])

ax[0].axhline(1/nA, linestyle=':')

ax[0].set_xlabel('$\delta$')
ax[1].set_xlabel('$\delta$')

ax[0].set_ylabel('$\gamma$')
ax[1].set_ylabel('$q$')

ax[0].legend()
ax[1].legend()

plt.show()