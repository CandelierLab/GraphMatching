'''
Star-branched graph: average gamma and q
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import project
from Graph import *
from  Comparison import *

os.system('clear')

# === Parameters ===========================================================

nRun = 1000

l_k = [2, 5, 10]
err_alpha = 0.2
ks = ['-', '--', ':']

# --------------------------------------------------------------------------

fname = project.root + f'/Files/Self-matching/SB/nRun={nRun:d}.csv'

# ==========================================================================

if os.path.exists(fname):

  # Load data
  df = pd.read_csv(fname)

  # Retrieve l_k, l_n and l_eta

  # l_k = np.unique(df.k)
  l_n = np.unique(df.n)
  # l_eta = np.unique(df.eta)



# --- Display --------------------------------------------------------------

fig, ax = plt.subplots(1,2, figsize=(12,6))

# Colors
c = {'GASM': '#1B2ACC', 'eGASM': '#089FFF',
     'Zager': '#CC4F1B', 'eZager': '#FF9848',
     'FAQ': '#3F7F4C', 'eFAQ':'#7EFF99'}

# --- Plots

for ki, k in enumerate(l_k):

  data = df.loc[df['k'] == k]

  # --- Accuracy

  ax[0].plot(data.n, data.g_GASM, '-', color=c['GASM'], linestyle=ks[ki], label=f'GASM')
  # ax[0].fill_between(data.n, data.g_GASM - data.g_GASM_std, data.g_GASM + data.g_GASM_std, alpha=err_alpha, facecolor=c['eGASM'])

  ax[0].plot(data.n, data.g_Zager, '-', color=c['Zager'], linestyle=ks[ki], label=f'Zager')
  # ax[0].fill_between(data.n, data.g_Zager - data.g_Zager_std, data.g_Zager + data.g_Zager_std,  alpha=err_alpha, facecolor=c['eZager'])

  ax[0].plot(data.n, data.g_FAQ, '-', color=c['FAQ'], linestyle=ks[ki], label=f'FAQ')
  # ax[0].fill_between(data.n, data.g_FAQ - data.g_FAQ_std, data.g_FAQ + data.g_FAQ_std, alpha=err_alpha, facecolor=c['eFAQ'])

  # ax[0].plot(l_h, np.exp(-l_h/2), '--', color='k', label='Th')

  # --- Structural quality

  ax[1].plot(data.n, data.q_GASM, '-', color=c['GASM'], linestyle=ks[ki], label=f'GASM')
  ax[1].fill_between(data.n, data.q_GASM - data.q_GASM_std, data.q_GASM + data.q_GASM_std, alpha=err_alpha, facecolor=c['eGASM'])

  ax[1].plot(data.n, data.q_Zager, '-', color=c['Zager'], linestyle=ks[ki], label=f'Zager')
  ax[1].fill_between(data.n, data.q_Zager - data.q_Zager_std, data.q_Zager + data.q_Zager_std,  alpha=err_alpha, facecolor=c['eZager'])

  ax[1].plot(data.n, data.q_FAQ, '-', color=c['FAQ'], linestyle=ks[ki], label=f'FAQ')
  ax[1].fill_between(data.n, data.q_FAQ - data.q_FAQ_std, data.q_FAQ + data.q_FAQ_std, alpha=err_alpha, facecolor=c['eFAQ'])

# ax[0].set_xscale('log')
ax[0].set_yscale('log')

ax[0].set_ylim([0, 1])
ax[1].set_ylim([0.85, 1])

ax[0].set_xlabel('n')
ax[1].set_xlabel('n')

ax[0].set_ylabel('$\gamma$')
ax[1].set_ylabel('$q$')

ax[0].legend()
ax[1].legend()

plt.show()