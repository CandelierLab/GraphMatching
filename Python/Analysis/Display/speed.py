import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import project

os.system('clear')

# === Parameters ===========================================================

l_nA = [10, 20, 50, 100, 200, 500]

# ==========================================================================

t = np.empty(len(l_nA))

for i, nA in enumerate(l_nA):

  fname = project.root + f'/Files/Speed/nA={nA}.csv'

  if os.path.exists(fname):

    # Load data
    T = pd.read_csv(fname, index_col=0)

    # Compute mean and std
    m = T.mean()
    s = T.std()
  
    t[i] = m[0]

# --- Display --------------------------------------------------------------

plt.style.use('dark_background')

fig, ax = plt.subplots()

ax.plot(l_nA, t, '.-')

ax.set_xlabel(r'$n_A$')
ax.set_ylabel(r't (s)')

ax.set_xscale('log')
ax.set_yscale('log')

plt.show()