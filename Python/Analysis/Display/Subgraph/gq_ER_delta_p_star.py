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
cm = plt.cm.jet(np.linspace(0,1,10))
fig, ax = plt.subplots(1, 2, figsize=(10,5))

nA = []
a = []
b = []

for i, fname in enumerate(os.listdir(dname)):

  # --- Extract parameters
  res = p.search(fname)
  if res is not None:
    nA_ = int(res.group(1))
    nRun_ = int(res.group(2))

    if nRun_!=nRun: continue

  # if nA_!=1000: continue

  # Random matchings
  # ax[0].axhline(y = 1/nA_, color = 'w', linestyle = ':')

  # --- Load data

  gamma = pd.read_csv(dname + fname, index_col=0)

  # x-values
  delta = np.array([float(i) for i in list(gamma)])

  # Compute mean and std
  m = gamma.mean()
  s = gamma.std()

  # --- Fit

  def ffun(x, a, b):
    # return (1-1/nA_)*np.exp(-(x**b)/a) + 1/nA_
    return (1-1/nA_)*np.exp(-(x)/a) + 1/nA_
    # return np.exp(-(x)/a) + 1/nA_

  popt, pcov = curve_fit(ffun, delta, m, p0=(5,1))

  nA.append(nA_)
  a.append(popt[0])
  # b.append(popt[1])

  # --- Display ------------------------------------------------------------

  ax[0].plot(delta, m, '.', color=cm[i], label=nA_)

  # Fit
  ax[0].plot(bin, ffun(bin, *popt), ':', color=cm[i])

  # ax.axhline(1/nA, color='w', linestyle='--')

  # g0 = -np.log(nA)
  # ax.plot(delta, -(np.log(m)-g0)/g0  , '.-', label=nA)

# --- Fit coefficients -----------------------------------------------------
 
I = np.argsort(nA)
nA = np.array(nA)[I]
a = np.array(a)[I]
# b = np.array(b)[I]
ax[1].plot(nA, a, '.-')
# ax[2].plot(nA, b, '.-')

# Fit of a
def affine(x, a, b):
    return a*x**b

popt, pcov = curve_fit(affine, nA, a)

print(popt)
ax[1].plot(nA, affine(nA, 0.5, -0.2), 'r-')


# --- Misc display settings ------------------------------------------------

ax[0].set_xscale('log')
# ax[0].set_yscale('log')

ax[0].set_xlabel('$\delta$')
ax[0].set_xlim(0, 1)
ax[0].set_ylabel('accuracy $\gamma$')

ax[1].set_xscale('log')
ax[1].set_yscale('log')

ax[1].set_xlabel('$n_A$')

ax[1].set_ylabel('$a$')

ax[0].legend()

plt.show()