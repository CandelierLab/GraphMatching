import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import project

os.system('clear')

# === Parameters ===========================================================

n = 100

# Average number of edges per node

nIter = 10

nRun = 100

# --------------------------------------------------------------------------

fname = project.root + '/Files/Normalization/ER/n={:d}_nIter={:d}_nRun={:d}.csv'.format(n, nIter, nRun)

# ==========================================================================

# Load data
F = pd.read_csv(fname, index_col=0)

# x-values
x = np.array([float(i) for i in list(F)])

# Compute mean and std
mv = F.mean()
s = F.std()

# Simple normalization factor
f0 = 4*x**2 + 1 

# f1 = 4*x**2 + 4*x**0.8 + 1 

# === Display =================================================================

# plt.style.use('dark_background')

fig, ax = plt.subplots()

ax.plot(x, f0, '--', color='k', linewidth=1)
ax.fill_between(x, mv-s, mv+s,
    alpha=0.5, facecolor='#dddddd')

ax.plot(x, mv, '.')

# ax.plot(x, f1, 'r-', linewidth=1)

ax.set_xlabel('Ratio of edge per node')
ax.set_ylabel('Normalization factor')

ax.set_xscale('log')
ax.set_yscale('log')

ax.set_title('n = {:d}'.format(n))

# ax.set_ylim(1, 6e4)

plt.show()