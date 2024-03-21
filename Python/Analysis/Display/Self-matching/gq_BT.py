'''
Balanced tree: average gamma and q
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

r = 2

# --------------------------------------------------------------------------

fname = project.root + f'/Files/Self-matching/BT/r={r:d}.csv'

# ==========================================================================

if os.path.exists(fname):

  # Load data
  df = pd.read_csv(fname)

  # Retrieve l_h l_eta

  l_h = np.unique(df.h)
  l_eta = np.unique(df.eta)

# --- Display --------------------------------------------------------------

plt.style.use('dark_background')
fig, ax = plt.subplots(1,2, figsize=(12,6))

# Colors
# cm = plt.cm.rainbow(np.linspace(0, 1, l_eta.size))

# --- Accuracy

g_FAQ = np.zeros(l_h.size)
q_FAQ = np.zeros(l_h.size)

g_Zager = np.zeros(l_h.size)
q_Zager = np.zeros(l_h.size)

for i, eta in enumerate(l_eta):

  data = df.loc[df['eta'] == eta]

  # Accuracy
  g_FAQ += data.g_FAQ.to_list()
  g_Zager += data.g_Zager.to_list()
  ax[0].plot(data.h, data.g_GASM, '-', label=f'$\eta = {eta:g}$')

  # Structural quality
  q_FAQ += data.q_FAQ.to_list()
  q_Zager += data.q_Zager.to_list()
  ax[1].plot(data.h, data.q_GASM, '-', label=f'$\eta = {eta:g}$')

ax[0].plot(l_h, g_FAQ/l_eta.size, '-', label='FAQ')
ax[1].plot(l_h, q_FAQ/l_eta.size, '-', label='FAQ')

ax[0].plot(l_h, g_Zager/l_eta.size, '-', label='Zager')
ax[1].plot(l_h, q_Zager/l_eta.size, '-', label='Zager')

ax[0].plot(l_h, np.exp(-l_h/2), '--', label='Th')

ax[0].set_yscale('log')

ax[0].set_ylim([0,1.01])
ax[1].set_ylim([0, 1.01])

ax[0].set_xlabel('h')
ax[1].set_xlabel('h')

ax[0].set_ylabel('$\gamma$')
ax[1].set_ylabel('$q$')

ax[0].legend()
ax[1].legend()

plt.show()