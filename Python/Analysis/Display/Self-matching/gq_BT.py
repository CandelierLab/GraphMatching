'''
Balanced tree: average gamma and q
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
r = 2

# --------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename', help='File to save the figure')
args = parser.parse_args()
figfile = args.filename

# ==========================================================================

if figfile is None:
  os.system('clear')

# --- Display --------------------------------------------------------------

plt.style.use('dark_background')
fig, ax = plt.subplots(1, 2, figsize=(20,10))

# Colors
# cm = plt.cm.rainbow(np.linspace(0, 1, l_eta.size))

for algo in l_algo:

  # --- Load data ----------------------------------------------------------

  datapath = project.root + f'/Files/Self-matching/BT/{algo}_r={r:d}.csv'

  if os.path.exists(datapath):

    # Load data
    data = pd.read_csv(datapath)

    # Retrieve l_h, l_eta and l_nRun
    l_h = np.unique(data.h)
    l_eta = np.unique(data.eta)

    # --- Plots ------------------------------------------------------------

    for i, eta in enumerate(l_eta):

      # Accuracy
      ax[0].plot(data.h, data.g, '-', label=algo)
      
      # Structural quality
      ax[1].plot(data.h, data.q, '-', label=algo)
    
ax[0].plot(l_h, np.exp(-l_h/2), '--', label='Th')

ax[0].set_yscale('log')

ax[0].set_ylim([1e-3, 1])
ax[1].set_ylim([0, 1.01])

ax[0].set_xlabel('h')
ax[1].set_xlabel('h')

ax[0].set_ylabel('$\gamma$')
ax[1].set_ylabel('$q$')

ax[0].legend()
ax[1].legend()

#  --- Output --------------------------------------------------------------

if figfile is None:

  plt.show()

else:

  print(f'Saving file {figfile} ...', end='', flush=True)
  tref = time.time()

  plt.savefig(project.root + '/Figures/' + figfile)

  print(' {:.02f} sec'.format((time.time() - tref)))