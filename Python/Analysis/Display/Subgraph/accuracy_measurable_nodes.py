import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import project

os.system('clear')

# === Parameters ===========================================================

nA = 15
nRun = 500

# --------------------------------------------------------------------------

fname = project.root + f'/Files/Subgraph/ER/Meas_nodes_nA={nA:d}_nRun={nRun:d}.csv'

# ==========================================================================

if os.path.exists(fname):

  # Load data
  df = pd.read_csv(fname)

  # Retrieve l_delta and l_zeta

  l_delta = np.unique(df.delta)
  l_zeta = np.unique(df.zeta)

# l_zeta = np.array([0])

# --- Display

plt.style.use('dark_background')
fig, ax = plt.subplots()

# Colors
cm = plt.cm.spring(np.linspace(0, 1, l_zeta.size))

for i, zeta in enumerate(l_zeta):

  data = df.loc[df['zeta'] == zeta]

  ax.plot((1-data.delta)*nA, data.g_Zager, '.-', label=f'$\zeta = {zeta}$')

  # ax.plot(data.delta, data.g_Zager, '--', color=cm[i], label=f'$\zeta = {zeta}$')
  # ax.plot(data.delta, data.g_GASM, '.-', color=cm[i], label=f'$\zeta = {zeta}$')

ax.set_ylim(0,1)

ax.legend()
ax.grid(True)

plt.show()