import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import project

os.system('clear')

# === Parameters ===========================================================

dname = project.root + '/Files/Success ratios/Meas_nodes_noise/'

refname = project.root + '/Files/Success ratios/Meas_nodes/ER_nA=100_zeta=0_nRun=1000.csv'

# ==========================================================================

# --- Get Parameters

# Regular expression
p = re.compile("ER_nA=(.*)_sigma=(.*)_nRun=(.*)\.csv")

nA = None
nRun = None

tmp = []

for fname in os.listdir(dname):

  # --- Extract parameters
  res = p.search(fname)
  if res is not None:
    if nA is None: nA = int(res.group(1))
    if nRun is None: nRun = int(res.group(3))
    tmp.append(float(res.group(2)))
    
l_sigma = np.sort(tmp)

# --- Display

fig, ax = plt.subplots()
ax.set_prop_cycle(plt.cycler(color=plt.cm.turbo(np.linspace(0,1,len(l_sigma)+1))))
                         
for sigma in l_sigma:
    
  # --- Load data

  fname = dname + '/ER_nA={:d}_sigma={:f}_nRun={:d}.csv'.format(nA, sigma, nRun)
  gamma = pd.read_csv(fname, index_col=0)

  # x-values
  Nsub = np.array([int(i) for i in list(gamma)])
  rho = Nsub/Nsub[-1]

  # Compute mean and std
  m = gamma.mean()
  s = gamma.std()

  # Plot
  ax.plot(rho, m, '.-', label=sigma)

# --- Reference

gamma = pd.read_csv(refname, index_col=0)

# x-values
Nsub = np.array([int(i) for i in list(gamma)])
rho = Nsub/Nsub[-1]

# Compute mean and std
m = gamma.mean()

# Plot
ax.plot(rho, m, 'k--', label='Ref')

# --- Misc display

ax.set_xlabel(r'subgraph ratio $\rho$')
ax.set_xlim(0, 1)
ax.set_ylabel(r'accuracy $\gamma$')
ax.set_ylim(0, 1)

ax.legend()

plt.show()