'''
-- Display --

Speed benchmark: time vs nA for ER graphs with p=log(n)/n

'''

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import project

os.system('clear')

# === Parameters ===========================================================

l_directed = [False, True]
l_algo = ['FAQ', '2opt', 'Zager', 'GASM_CPU', 'GASM_GPU']

lw = 3
fontsize = 24

# Colors
c = {'2opt': '#CC4F1B', 'e2opt': '#FF9848',
     'FAQ': '#FFA500', 'eFAQ': '#FACC2E',
     'Zager': '#1B2ACC', 'eZager': '#089FFF',
     'GASM_CPU': '#3F7F4C', 'eGASM':'#7EFF99',
     'GASM_GPU': '#800080', 'eGASM':'#7EFF99',}
# --------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename', help='File to save the figure')
args = parser.parse_args()
figfile = args.filename

# ==========================================================================

# Prepare figure
plt.rcParams.update({'font.size': fontsize})
fig, ax = plt.subplots(1,2, figsize=[20,10])

for directed in l_directed:

  sdir = 'directed' if directed else 'undirected'

  for algo in l_algo:

    fname = project.root + f'/Files/Speed/n/ER_{algo}_{sdir:s}.csv'

    if os.path.exists(fname):

      # Load data
      df = pd.read_csv(fname)

      # Retrieve l_n
      l_n = np.unique(df.n)

      data = df.groupby('n')['t'].mean().to_frame()

      # --- Display --------------------------------------------------------------

      if directed:
        ax[1].plot(l_n, data.t, color=c[algo], linewidth=lw)
      else:
        ax[0].plot(l_n, data.t, color=c[algo], linewidth=lw, label=algo)

ax[0].set_title('undirected')
ax[0].set_xlabel('$n_A$')
ax[0].set_ylabel('$t$ (ms)')

ax[0].set_xlim(1, 1e4)
ax[0].set_ylim(0.01, 2e5)
ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].set_box_aspect(1)

ax[0].legend()
ax[0].grid(True)

ax[1].set_title('directed')
ax[1].set_xlabel('$n_A$')
ax[1].set_ylabel('$t$ (ms)')

ax[1].set_xlim(1, 1e4)
ax[1].set_ylim(0.01, 2e5)
ax[1].set_xscale('log')
ax[1].set_yscale('log')
ax[1].set_box_aspect(1)

ax[1].grid(True)

plt.show()