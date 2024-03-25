'''
Erdo-Renyi: average gamma and q
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import project

os.system('clear')

# === Parameters ===========================================================

nA = 20
nRun = 100

err_alpha = 0.2

# --------------------------------------------------------------------------

fname = project.root + f'/Files/Self-matching/ER/nA={nA:d}_nRun={nRun:d}.csv'

# ==========================================================================

if os.path.exists(fname):

  # Load data
  df = pd.read_csv(fname)

  # Retrieve l_p and l_eta

  l_p = np.unique(df.p)
  l_eta = np.unique(df.eta)

data = df.loc[df['eta'] == l_eta[2]]

# --- Display --------------------------------------------------------------

fig, ax = plt.subplots(1,2, figsize=(12,6))

# Colors
c = {'GASM': '#1B2ACC', 'eGASM': '#089FFF',
     'Zager': '#CC4F1B', 'eZager': '#FF9848',
     'FAQ': '#3F7F4C', 'eFAQ':'#7EFF99'}

# --- Accuracy

ax[0].plot(data.p, data.g_GASM, '-', color=c['GASM'], label=f'GASM')
# ax[0].fill_between(data.h, data.g_GASM - data.g_GASM_std, data.g_GASM + data.g_GASM_std, alpha=err_alpha, facecolor=c['eGASM'])

ax[0].plot(data.p, data.g_Zager, '-', color=c['Zager'], label=f'Zager')
# ax[0].fill_between(data.h, data.g_Zager - data.g_Zager_std, data.g_Zager + data.g_Zager_std,  alpha=err_alpha, facecolor=c['eZager'])

ax[0].plot(data.p, data.g_FAQ, '-', color=c['FAQ'], label=f'FAQ')
# ax[0].fill_between(data.h, data.g_FAQ - data.g_FAQ_std, data.g_FAQ + data.g_FAQ_std, alpha=err_alpha, facecolor=c['eFAQ'])

# --- Structural quality

ax[1].plot(data.p, data.q_GASM, '-', color=c['GASM'], label=f'GASM')
ax[1].fill_between(data.p, data.q_GASM - data.q_GASM_std, data.q_GASM + data.q_GASM_std, alpha=err_alpha, facecolor=c['eGASM'])

ax[1].plot(data.p, data.q_Zager, '-', color=c['Zager'], label=f'Zager')
ax[1].fill_between(data.p, data.q_Zager - data.q_Zager_std, data.q_Zager + data.q_Zager_std,  alpha=err_alpha, facecolor=c['eZager'])

ax[1].plot(data.p, data.q_FAQ, '-', color=c['FAQ'], label=f'FAQ')
ax[1].fill_between(data.p, data.q_FAQ - data.q_FAQ_std, data.q_FAQ + data.q_FAQ_std, alpha=err_alpha, facecolor=c['eFAQ'])


ax[0].set_ylim([0, 1.01])
ax[1].set_ylim([0.9, 1.001])

ax[0].set_xlabel('$n$')
ax[1].set_xlabel('$n$')

ax[0].set_ylabel('$\gamma$')
ax[1].set_ylabel('$q_s$')

ax[0].legend()
ax[1].legend()

plt.show()