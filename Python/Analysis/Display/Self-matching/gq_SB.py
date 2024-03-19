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

nRun = 100

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

plt.style.use('dark_background')
fig, ax = plt.subplots(1,2, figsize=(12,6))

# Colors
cm = plt.cm.gist_rainbow(np.linspace(0, 1, l_k.size))

# --- Plots

for ki, k in enumerate(l_k):

  g_Zager = np.zeros(l_n.size)
  q_Zager = np.zeros(l_n.size)

  data = df.loc[df['k'] == k]

  # Accuracy    
  ax[0].plot(data.n, data.g_GASM, '-', color=cm[ki], label=f'GASM $k = {k:d}$')
  ax[0].plot(data.n, data.g_Zager, '--', color=cm[ki], label=f'Zager $k = {k:d}$')

  # Structural quality  
  ax[1].plot(data.n, data.q_GASM, '-', color=cm[ki], label=f'GASM $k = {k:d}$')
  ax[1].plot(data.n, data.q_Zager, '--', color=cm[ki], label=f'Zager $k = {k:d}$')

ax[0].set_ylim([0,1.01])
ax[1].set_ylim([0, 1.01])

ax[0].set_xlabel('n')
ax[1].set_xlabel('n')

ax[0].set_ylabel('$\gamma$')
ax[1].set_ylabel('$q$')

ax[0].legend()
ax[1].legend()

ax[1].grid(True)

plt.show()