'''
Circular ladder graph: average gamma and q
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import project
from Network import *
from  Comparison import *

os.system('clear')

# === Parameters ===========================================================

# --------------------------------------------------------------------------

fname = project.root + f'/Files/Self-matching/CL/CL.csv'

# ==========================================================================

if os.path.exists(fname):

  # Load data
  df = pd.read_csv(fname)

  # Retrieve l_n and l_eta

  l_n = np.unique(df.n)
  l_eta = np.unique(df.eta)

# --- Display --------------------------------------------------------------

plt.style.use('dark_background')
fig, ax = plt.subplots(1,2, figsize=(12,6))

# --- Accuracy

g_FAQ = np.zeros(l_n.size)
q_FAQ = np.zeros(l_n.size)

g_Zager = np.zeros(l_n.size)
q_Zager = np.zeros(l_n.size)

for i, eta in enumerate(l_eta):

  data = df.loc[df['eta'] == eta]

  # Accuracy
  g_FAQ += data.g_FAQ.to_list()
  g_Zager += data.g_Zager.to_list()
  ax[0].plot(data.n, data.g_GASM, '.', label=f'$\eta = {eta:g}$')

  # Structural quality
  q_FAQ += data.q_FAQ.to_list()
  q_Zager += data.q_Zager.to_list()
  ax[1].plot(data.n, data.q_GASM, '.-', label=f'$\eta = {eta:g}$')

ax[0].plot(l_n, g_FAQ/l_eta.size, '.', label='FAQ')
ax[1].plot(l_n, q_FAQ/l_eta.size, '.-', label='FAQ')

ax[0].plot(l_n, g_Zager/l_eta.size, '.', label='Zager')
ax[1].plot(l_n, q_Zager/l_eta.size, '.-', label='Zager')

ax[0].plot(l_n, 0.5/l_n, '--', color='white', label='$\\frac{1}{2n}$')

# Plot 0 scales 
# ax[0].set_yscale('log')
# ax[0].set_xscale('log')
ax[0].set_xlim([0, np.max(l_n)+1])
ax[0].set_ylim([0, 1.01])

# Plot 1 scales
ax[1].set_xlim([0, np.max(l_n)+1])
ax[1].set_ylim([0, 1.01])

ax[0].set_xlabel('n')
ax[1].set_xlabel('n')

ax[0].set_ylabel('$\gamma$')
ax[1].set_ylabel('$q$')

ax[0].legend()
ax[1].legend()

plt.show()