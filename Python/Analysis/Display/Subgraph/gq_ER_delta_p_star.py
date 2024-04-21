import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import project

os.system('clear')

# === Parameters ===========================================================

dname = project.root + '/Files/Subgraph/delta/'
nRun = 10000

# ==========================================================================

# --- Fit

# def ffun(x, a, b):
#   return (1-b)*np.exp(-x/a) + b

bin = np.linspace(0,1,101)

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
    nRun_ = int(res.group(2))

    if nRun_!=nRun: continue

  # Random matchings
  # ax.axhline(y = 1/nA, color = 'w', linestyle = ':')

  # --- Load data

  gamma = pd.read_csv(dname + fname, index_col=0)

  # x-values
  delta = np.array([float(i) for i in list(gamma)])

  # Compute mean and std
  m = gamma.mean()
  s = gamma.std()

  # --- Fit

  def ffun(x, a, b):
    return (1-1/nA)*np.exp(-(x**b)/a) + 1/nA

  popt, pcov = curve_fit(ffun, delta, m, p0=(5,1))

  # --- Display ------------------------------------------------------------

  ax.plot(delta, m, '.', label=nA)

  # Fit
  ax.plot(bin, ffun(bin, *popt), 'w:')

  # ax.axhline(1/nA, color='w', linestyle='--')

  # g0 = -np.log(nA)
  # ax.plot(delta, -(np.log(m)-g0)/g0  , '.-', label=nA)
  
  
# --- Misc display settings ------------------------------------------------

ax.set_xscale('log')
ax.set_yscale('log')

ax.set_xlabel('$\delta$')
ax.set_xlim(0, 1)
ax.set_ylabel('accuracy $\gamma$')

ax.legend()

plt.show()