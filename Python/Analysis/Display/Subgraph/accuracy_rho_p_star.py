import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import project

# os.system('clear')

# === Parameters ===========================================================

dname = project.root + '/Files/Success ratios/rho/'

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
  Nsub = np.array([int(i) for i in list(gamma)])
  rho = Nsub/Nsub[-1]

  # Compute mean and std
  m = gamma.mean()
  s = gamma.std()

  # Approx
  # y = np.exp((rho-1)*np.log(nA))

  # --- Display

  ax.plot(rho, m, '.-', label=nA)
  # ax.plot(rho, y, 'w:')

  # ax.plot(rho, m/y, '.-')

# --- Misc display settings

ax.set_yscale('log')

ax.set_xlabel(r'subgraph ratio $\rho$')
ax.set_xlim(0, 1)
ax.set_ylabel(r'Accuracy $\gamma$')
# ax.set_ylim(0, 1.2)

ax.legend()

plt.show()