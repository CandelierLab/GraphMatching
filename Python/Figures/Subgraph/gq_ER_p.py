'''
-- Display --

Subgraph degradation
Erdo-Renyi: average gamma and q as a function of p
'''

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import project

# === Parameters ===========================================================

directed = True
nA = 20

# nRun = 10000
nRun = 100

# --- Plot parameters

err_alpha = 0.2
lw = 3
fontsize = 16

# --------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename', help='File to save the figure')
args = parser.parse_args()
figfile = args.filename

# --------------------------------------------------------------------------

datapath = project.root + f'/Files/Subgraph/ER/ER_nA={nA:d}_nRun={nRun:d}.csv'

# ==========================================================================

if figfile is None:
  os.system('clear')

if os.path.exists(datapath):

  # Load data
  df = pd.read_csv(datapath)

  # Retrieve l_p and l_delta

  l_p = np.unique(df.p)
  l_delta = np.unique(df.delta)

# --- Display --------------------------------------------------------------

# Colors
cm = plt.cm.jet(np.linspace(0, 1, l_delta.size))

plt.rcParams.update({'font.size': fontsize})
fig, ax = plt.subplots(1, 1, figsize=(5,5))

for i, delta in enumerate(l_delta):

  data = df.loc[np.logical_and(df['directed'] == directed, df['delta'] == delta)]

  # ax.plot(data.p, data.g_Zager, '--', color=cm[i])
  ax.plot(data.p, data.g_GASM, '-', color=cm[i], label=f'$\delta = {delta:g}$')

ax.axvline(2/nA, color='k', linestyle='--')

# --- Misc figure settings

ax.set_xlabel('p')
ax.set_ylabel('$\gamma$')

ax.set_xlim(0, 1)
ax.set_yscale('log')
ax.legend()

plt.show()