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

nA = 100
nRun = 10

# --------------------------------------------------------------------------

fname = project.root + f'/Files/Degradation/ER/nA={nA:d}_nRun={nRun:d}.csv'

# ==========================================================================

if os.path.exists(fname):

  # Load data
  df = pd.read_csv(fname)

  # Retrieve l_delta and l_meas

  l_delta = np.unique(df.delta)
  l_meas = np.unique(df.nMeasAttr)

# --- Display --------------------------------------------------------------

plt.style.use('dark_background')
fig, ax = plt.subplots(1,2)

# --- FAQ

data = df.loc[df['algo'] == 'FAQ']
ax[0].plot(l_delta, data.g, '--', color='w', label=f'FAQ')
ax[1].plot(l_delta, data.q, '--', color='w', label=f'FAQ')

# --- GASM

for m in l_meas:
  data = df.loc[np.logical_and(df['algo']=='GASM', df['nMeasAttr']==m)]
  ax[0].plot(l_delta, data.g, '.-', label=f'GASM {m}')
  ax[1].plot(l_delta, data.q, '.-', label=f'GASM {m}')

ax[0].set_yscale('log')

ax[0].set_ylim([0.005, 1])
ax[1].set_ylim([0.9, 1])

ax[0].axhline(1/nA, linestyle=':')

ax[0].set_xlabel('$\delta$')
ax[1].set_xlabel('$\delta$')

ax[0].set_ylabel('$\gamma$')
ax[1].set_ylabel('$q$')

ax[0].legend()
ax[1].legend()

plt.show()