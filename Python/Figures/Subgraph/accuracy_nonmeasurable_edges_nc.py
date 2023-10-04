import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import project

os.system('clear')

# === Parameters ===========================================================

dname = project.root + '/Files/Success ratios/nMeas_edges_nc/'

# ==========================================================================

# Regular expression
p = re.compile("ER_nA=(.*)_nc=(.*)_nRun=(.*)\.csv")

# --- Display

fig, ax = plt.subplots()

for fname in os.listdir(dname):

  # --- Extract parameters
  res = p.search(fname)
  if res is not None:
    nA = int(res.group(1))
    nc = int(res.group(2))
    nRun = int(res.group(3))

  # --- Load data

  gamma = pd.read_csv(dname + fname, index_col=0)

  # x-values
  Nsub = np.array([int(i) for i in list(gamma)])
  rho = Nsub/Nsub[-1]

  # Compute mean and std
  m = gamma.mean()
  s = gamma.std()

  # Plot
  ax.plot(rho, m, '.-', label=nc)

# --- Misc display

# Random matchings

ax.set_xlabel(r'subgraph ratio $\rho$')
ax.set_xlim(0, 1)
ax.set_ylabel(r'Accuracy $\gamma$')
ax.set_ylim(0, 1)

ax.legend()

plt.show()