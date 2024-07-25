import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import project

os.system('clear')

# === Parameters ===========================================================

directed = True
nA = 20
nRun = 1000

# --------------------------------------------------------------------------

ds = 'directed' if directed else 'undirected'

fname = project.root + f'/Files/Subgraph/ER/{ds}_edges_measurable_nA={nA:d}_nRun={nRun:d}.csv'

# ==========================================================================

if os.path.exists(fname):

  # Load data
  df = pd.read_csv(fname)

  # Retrieve l_delta and l_xi_m

  l_delta = np.unique(df.delta)
  l_xi_m = np.unique(df.xi_m)

# === Display ==============================================================

plt.style.use('dark_background')
fig, ax = plt.subplots(1, 2, figsize=(20,10))

# Colors
cm = plt.cm.jet(np.linspace(0, 1, l_xi_m.size))

# --- Zager

data = df.loc[df['xi_m'] == 0]

ax[0].plot(data.delta, data.g_Zager, 'w--', label=f'Zager')
ax[1].plot(data.delta, data.q_Zager, 'w--', label=f'Zager')

# --- GASM

for i, xi_m in enumerate(l_xi_m):

  data = df.loc[df['xi_m'] == xi_m]

  ax[0].plot(data.delta, data.g_GASM, '.-', color=cm[i], label=f'GASM $\\xi_m = {xi_m}$')
  ax[1].plot(data.delta, data.q_GASM, '.-', color=cm[i], label=f'GASM $\\xi_m = {xi_m}$')

# Axes limits
ax[0].set_ylim(0,1)
ax[0].set_xlim(0,1)
ax[1].set_xlim(0,1)
ax[1].set_ylim(0,1)

ax[0].set_xlabel(r'$\delta$')
ax[0].set_ylabel(r'$\gamma$')

ax[1].set_xlabel(r'$\delta$')
ax[1].set_ylabel(r'$q_s$')

ax[0].legend()
ax[1].legend()

ax[0].grid(True)
ax[1].grid(True)

plt.show()