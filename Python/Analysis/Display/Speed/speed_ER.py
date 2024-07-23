import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import project

os.system('clear')

# === Parameters ===========================================================

l_directed = [False, True]
l_algo = ['FAQ', '2opt', 'Zager', 'GASM_CPU']

# ==========================================================================

# Prepare figure
plt.style.use('dark_background')
fig, ax = plt.subplots()


for directed in l_directed:

  sdir = 'directed' if directed else 'undirected'

  for algo in l_algo:

    fname = project.root + f'/Files/Speed/ER_{algo}_{sdir:s}.csv'

    if os.path.exists(fname):

      # Load data
      df = pd.read_csv(fname)

      # Retrieve l_n
      l_n = np.unique(df.n)

      data = df.groupby('n')['t'].mean().to_frame()

      print(data)

      # --- Display --------------------------------------------------------------

      if directed:
        ax.plot(l_n, data.t, '--', label=algo)
      else:
        ax.plot(l_n, data.t, '-', label=algo)

ax.set_xlabel('$n_A$')
ax.set_ylabel('$t$ (ms)')

ax.set_xscale('log')
ax.set_yscale('log')

ax.legend()
ax.grid(True)

plt.show()