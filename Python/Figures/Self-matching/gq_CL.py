'''
Circular ladder graph: average gamma and q
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

err_alpha = 0.2

# --------------------------------------------------------------------------

fname = project.root + f'/Files/Self-matching/CL/CL.csv'

# ==========================================================================

if os.path.exists(fname):

  # Load data
  df = pd.read_csv(fname)

  # Retrieve l_h

  l_n = np.unique(df.n)

print(df)

# --- Display --------------------------------------------------------------

fig, ax = plt.subplots(1,2, figsize=(12,6))

# Colors
c = {'GASM': '#1B2ACC', 'eGASM': '#089FFF',
     'Zager': '#CC4F1B', 'eZager': '#FF9848',
     'FAQ': '#3F7F4C', 'eFAQ':'#7EFF99'}

# --- Accuracy

ax[0].plot(df.n, df.g_GASM, '-', color=c['GASM'], label=f'GASM')
# ax[0].fill_between(df.h, df.g_GASM - df.g_GASM_std, df.g_GASM + df.g_GASM_std, alpha=err_alpha, facecolor=c['eGASM'])

ax[0].plot(df.n, df.g_Zager, '-', color=c['Zager'], label=f'Zager')
# ax[0].fill_between(df.h, df.g_Zager - df.g_Zager_std, df.g_Zager + df.g_Zager_std,  alpha=err_alpha, facecolor=c['eZager'])

ax[0].plot(df.n, df.g_FAQ, '-', color=c['FAQ'], label=f'FAQ')
# ax[0].fill_between(df.h, df.g_FAQ - df.g_FAQ_std, df.g_FAQ + df.g_FAQ_std, alpha=err_alpha, facecolor=c['eFAQ'])

ax[0].plot(l_n, 1/2/l_n, '--', color='k', label='$\\frac{1}{2n}$')

# --- Structural quality

ax[1].plot(df.n, df.q_GASM, '-', color=c['GASM'], label=f'GASM')
ax[1].fill_between(df.n, df.q_GASM - df.q_GASM_std, df.q_GASM + df.q_GASM_std, alpha=err_alpha, facecolor=c['eGASM'])

ax[1].plot(df.n, df.q_Zager, '-', color=c['Zager'], label=f'Zager')
ax[1].fill_between(df.n, df.q_Zager - df.q_Zager_std, df.q_Zager + df.q_Zager_std,  alpha=err_alpha, facecolor=c['eZager'])

ax[1].plot(df.n, df.q_FAQ, '-', color=c['FAQ'], label=f'FAQ')
ax[1].fill_between(df.n, df.q_FAQ - df.q_FAQ_std, df.q_FAQ + df.q_FAQ_std, alpha=err_alpha, facecolor=c['eFAQ'])


ax[0].set_xscale('log')
ax[0].set_yscale('log')

ax[0].set_ylim([0,1])
ax[1].set_ylim([0, 1])

ax[0].set_xlabel('$n$')
ax[1].set_xlabel('$n$')

ax[0].set_ylabel('$\gamma$')
ax[1].set_ylabel('$q_s$')

ax[0].legend()
ax[1].legend()

plt.show()