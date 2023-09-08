import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import project

os.system('clear')

# === Parameters ===========================================================

n = 100

# Average number of edges per node
# l_nepn = [0.1, 1, 10, 50]
l_nepn = np.geomspace(0.1, 100, 20)

nIter = 10

nRun = 100

# --------------------------------------------------------------------------

fname = project.root + '/Files/Normalization/ER/n={:d}_nIter={:d}_nRun={:d}.csv'.format(n, nIter, nRun)

# ==========================================================================

# Load data
F = pd.read_csv(fname, index_col=0)

# x-values
x = np.array([float(i) for i in list(F)])/n

# Compute mean and std
mv = F.mean()
s = F.std()

# Simple normalisation factor
f0 = 4*(x*n)**2

# f1 = 4*(x*n)**2 + 18*(x*n)**0.7

m = x*n**2
f1 = 4*m**2/(n * (1-n*np.exp(-2*m/n)))**2

# === Display =================================================================

# plt.style.use('dark_background')

fig, ax = plt.subplots()

ax.plot(x, f0, '--', color='k', linewidth=1)
ax.fill_between(x, mv-s, mv+s,
    alpha=0.5, facecolor='#dddddd')

ax.plot(x, mv, '.')

ax.plot(x, f1, 'r-', linewidth=1)

ax.set_xlabel('Relative ratio of edge per node')
ax.set_ylabel('Normalization factor')

ax.set_xscale('log')
ax.set_yscale('log')

plt.show()