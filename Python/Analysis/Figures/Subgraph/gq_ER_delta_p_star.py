import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import project

os.system('clear')

# === Parameters ===========================================================

l_nA = [10, 20, 50, 100, 200, 500, 1000]
nRun = 10000

dname = project.root + '/Files/Subgraph/delta/'

# --- Plot parameters

err_alpha = 0.2
lw = 3
fontsize = 16

# ==========================================================================

bin = np.linspace(0,1,101)

# --- Display

cm = plt.cm.jet(np.linspace(0, 1, len(l_nA)))
plt.rcParams.update({'font.size': fontsize})
fig, ax = plt.subplots(1, 1, figsize=(5,5))

nA = []
a = []
b = []

for i, nA in enumerate(l_nA):

  # --- Load data

  fname = dname + f'ER_nA={nA}_nRun={nRun}.csv'

  gamma = pd.read_csv(fname, index_col=0)

  # x-values
  delta = np.array([float(i) for i in list(gamma)])

  # Compute mean and std
  m = gamma.mean()
  s = gamma.std()

  # --- Fit

  def ffun(x, a, b):
    return (1-1/nA)*np.exp(-(x)/a) + 1/nA

  popt, pcov = curve_fit(ffun, delta, m, p0=(5,1))

  # --- Display ------------------------------------------------------------

  ax.plot(delta, m, '.', color=cm[i], label=nA)

  # Fit
  ax.plot(bin, ffun(bin, *popt), ':', color=cm[i])

# --- Misc display settings ------------------------------------------------

ax.set_xscale('log')
# ax.set_yscale('log')

ax.set_xlabel('$\delta$')
ax.set_xlim(0.009, 1)
ax.set_ylim(0, 1)
ax.set_ylabel('accuracy $\gamma$')

ax.legend()

plt.show()