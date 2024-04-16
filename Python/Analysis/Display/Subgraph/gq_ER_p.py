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

nA = 20
nRun = 100

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

  # Retrieve l_directed, l_p and l_delta

  l_directed = np.unique(df.directed)
  l_p = np.unique(df.p)
  l_delta = np.unique(df.delta)

# --- Display --------------------------------------------------------------

plt.style.use('dark_background')
fig, ax = plt.subplots(2, 2, figsize=(20,10))

# Colors
cm = plt.cm.gist_rainbow(np.linspace(0, 1, l_delta.size))

for d in l_directed:

  for i, delta in enumerate(l_delta):

    data = df.loc[np.logical_and(df['directed'] == d, df['delta'] == delta)]

    ax[int(d),0].plot(data.p, data.g_Zager, '.--', color=cm[i])
    ax[int(d),0].plot(data.p, data.g_GASM, '.-', color=cm[i], label=f'$\delta = {delta:g}$')

    ax[int(d),1].plot(data.p, data.q_Zager, '--', color=cm[i])
    ax[int(d),1].plot(data.p, data.q_GASM, '-', color=cm[i], label=f'$\delta = {delta:g}$')

# for k in range(len(rho)):

#   ax.plot(l_p, Z[:,k], '.-')

#   S += Z[:,k]/len(l_p)

# p_star = l_p[np.argmax(S)]

# print(p_star)

# ax.axvline(p_star, color='w', linestyle='--')

# --- Titles

ax[0,0].set_title('Undirected')
ax[0,1].set_title('Undirected')
ax[1,0].set_title('Directed')
ax[1,1].set_title('Directed')

# --- Labels

ax[0,0].set_xlabel('p')
ax[0,0].set_ylabel('$\gamma$')

ax[0,1].set_xlabel('p')
ax[0,1].set_ylabel('$q$')

ax[1,0].set_xlabel('p')
ax[1,0].set_ylabel('$\gamma$')

ax[1,1].set_xlabel('p')
ax[1,1].set_ylabel('$q$')

# --- Scales and limits

ax[0,0].set_xlim(0, 1)
ax[0,1].set_xlim(0, 1)
ax[1,0].set_xlim(0, 1)
ax[1,1].set_xlim(0, 1)

# ax[0,0].set_yscale('log')
# ax[1,0].set_yscale('log')

# --- Misc display featutes (legend, grid, ...)

ax[0,1].legend()

plt.show()