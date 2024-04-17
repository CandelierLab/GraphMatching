import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import project

# os.system('clear')

# === Parameters ===========================================================

dname = project.root + '/Files/Subgraph/delta/'

# ==========================================================================

# Regular expression
p = re.compile("ER_nA=(.*)_nRun=(.*)\.csv")

# --- Display

plt.style.use('dark_background')
fig, ax = plt.subplots()

for fname in os.listdir(dname):

  # --- Extract parameters
  res = p.search(fname)
  if res is not None:
    nA = int(res.group(1))
    nRun = int(res.group(2))

  # Random matchings
  # ax.axhline(y = 1/nA, color = 'w', linestyle = ':')

  # --- Load data

  gamma = pd.read_csv(dname + fname, index_col=0)

  # x-values
  delta = np.array([float(i) for i in list(gamma)])

  # Compute mean and std
  m = gamma.mean()
  s = gamma.std()

  # --- Display

  # ax.plot(delta, np.log(m), '.-', label=nA)
  # ax.axhline(-np.log(nA), color='w', linestyle='--')

  g0 = -np.log(nA)
  ax.plot(delta, -(np.log(m)-g0)/g0  , '.-', label=nA)
  
  
# --- Misc display settings

# ax.set_xscale('log')
ax.set_yscale('log')

# ax.set_xlabel('$\delta$')
# ax.set_xlim(0, 1)
# ax.set_ylabel('accuracy $\gamma$')
# ax.set_ylim(0, 1.2)

ax.legend()

plt.show()