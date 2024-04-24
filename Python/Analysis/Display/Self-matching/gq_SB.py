'''
Star-branched graph: average gamma and q
'''

import os
import argparse
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

import project

# === Parameters ===========================================================

l_algo = ['FAQ', '2opt', 'Zager', 'GASM']

directed = False

ls = {'FAQ': '--', '2opt': '-.', 'Zager': ':', 'GASM':'-'}

# --------------------------------------------------------------------------

ds = 'directed' if directed else 'undirected'

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename', help='File to save the figure')
args = parser.parse_args()
figfile = args.filename

# ==========================================================================

if figfile is None:
  os.system('clear')

plt.style.use('dark_background')
fig, ax = plt.subplots(1,2, figsize=(20,10))

for algo in l_algo:

  fname = project.root + f'/Files/Self-matching/SB/{algo}_{ds}.csv'
  
  # print(fname)

  if os.path.exists(fname):

    # Load data
    df = pd.read_csv(fname)

    # Retrieve l_k and l_n

    l_k = np.unique(df.k).astype(int)
    l_n = np.unique(df.n).astype(int)

    # Colors
    cm = plt.cm.gist_rainbow(np.linspace(0, 1, l_k.size))

    # --- Plots

    for ki, k in enumerate(l_k):

      data = df.loc[df['k'] == k]

      # Accuracy    
      ax[0].plot(data.n, data.g, linestyle=ls[algo], color=cm[ki], label=f'{algo} $k = {k:d}$')

      # Structural quality  
      ax[1].plot(data.n, data.q, linestyle=ls[algo], color=cm[ki], label=f'{algo} $k = {k:d}$')

ax[0].set_ylim([0, 1])
ax[1].set_ylim([0.9, 1])

ax[0].set_xlabel('n')
ax[1].set_xlabel('n')

ax[0].set_ylabel('$\gamma$')
ax[1].set_ylabel('$q$')

ax[0].legend()
ax[1].legend()

ax[1].grid(True)

#  --- Output --------------------------------------------------------------

if figfile is None:

  plt.show()

else:

  print(f'Saving file {figfile} ...', end='', flush=True)
  tref = time.time()

  plt.savefig(project.root + '/Figures/' + figfile)

  print(' {:.02f} sec'.format((time.time() - tref)))