'''
Star-branched graph: average gamma and q
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import project
from Network import *
from  Comparison import *

os.system('clear')

# === Parameters ===========================================================

nRun = 1000

# --------------------------------------------------------------------------

fname = project.root + f'/Files/Self-matching/SB/nRun={nRun:d}.csv'

# ==========================================================================

if os.path.exists(fname):

  # Load data
  df = pd.read_csv(fname)

  # Retrieve l_k, l_n and l_eta

  l_k = np.unique(df.k)
  l_n = np.unique(df.n)
  l_eta = np.unique(df.eta)

# --- Display --------------------------------------------------------------

fig, ax = plt.subplots(1,2, figsize=(12,6))

# --- Plots

for ki, k in enumerate(l_k):

  data = df.loc[df['k'] == k]

  # Accuracy    
  ax[0].plot(data.n, data.g_GASM, '-', label=f'GASM $k = {k:d}$')
  ax[0].plot(data.n, data.g_FAQ, ':', label=f'FAQ $k = {k:d}$')
  ax[0].plot(data.n, data.g_Zager, '--', label=f'Zager $k = {k:d}$')

  # Structural quality  
  ax[1].plot(data.n, data.q_GASM, '-', label=f'GASM $k = {k:d}$')
  ax[1].plot(data.n, data.q_FAQ, ':', label=f'FAQ $k = {k:d}$')
  ax[1].plot(data.n, data.q_Zager, '--', label=f'Zager $k = {k:d}$')

ax[0].set_ylim([0, 1.01])
ax[1].set_ylim([0, 1.01])

ax[0].set_xlabel('n')
ax[1].set_xlabel('n')

ax[0].set_ylabel('$\gamma$')
ax[1].set_ylabel('$q$')

ax[0].legend()
ax[1].legend()

ax[1].grid(True)

plt.show()