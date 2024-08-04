import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import project

os.system('clear')

# === Parameters ===========================================================

l_directed = [False, True]
l_algo = ['FAQ', '2opt', 'Zager', 'GASM_CPU', 'GASM_GPU']

# ==========================================================================

# Prepare figure
plt.style.use('dark_background')
fig, ax = plt.subplots(1,2, figsize=[10,5])


for directed in l_directed:

  sdir = 'directed' if directed else 'undirected'

  for algo in l_algo:

    fname = project.root + f'/Files/Speed/n/ER_{algo}_{sdir:s}.csv'

    if os.path.exists(fname):

      # Load data
      df = pd.read_csv(fname)

      # Retrieve l_n
      l_n = np.unique(df.n)

      data = df.groupby('n')['t'].mean().to_frame()

      # --- Display --------------------------------------------------------------

      if directed:
        ax[1].plot(l_n, data.t, '-', label=algo)
      else:
        ax[0].plot(l_n, data.t, '-', label=algo)

ax[0].set_title('Undirected')
ax[0].set_xlabel('$n_A$')
ax[0].set_ylabel('$t$ (ms)')

ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].set_box_aspect(1)

ax[0].legend()
ax[0].grid(True)

ax[1].set_title('Directed')
ax[1].set_xlabel('$n_A$')
ax[1].set_ylabel('$t$ (ms)')

ax[1].set_xscale('log')
ax[1].set_yscale('log')
ax[1].set_box_aspect(1)

ax[1].legend()
ax[1].grid(True)

plt.show()