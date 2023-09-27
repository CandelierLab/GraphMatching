import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import project

# os.system('clear')

# === Parameters ===========================================================

nA = 100
p = np.log(nA)/nA
nRun = 1000

# --------------------------------------------------------------------------

fname = project.root + '/Files/Success ratios/ER_p={:.02f}_nA={:d}_nRun={:d}.csv'.format(p, nA, nRun)

# ==========================================================================

# Load data
gamma = pd.read_csv(fname, index_col=0)

# x-values
Nsub = np.array([int(i) for i in list(gamma)])
rho = Nsub/Nsub[-1]

# Compute mean and std
m = gamma.mean()
s = gamma.std()

# === Display =================================================================

plt.style.use('dark_background')

fig, ax = plt.subplots()

ax.axhline(y = 1/nA, color = 'w', linestyle = ':')

ax.plot(rho, m)

ax.set_xlabel(r'subgraph ratio $\rho$')
ax.set_xlim(0, 1)
ax.set_ylabel(r'ratio of correct matches $\gamma$')
ax.set_ylim(0, 1)

plt.show()