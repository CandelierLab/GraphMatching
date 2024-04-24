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

l_h = None

for algo in l_algo:

  # --- Load data ----------------------------------------------------------

  datapath = project.root + f'/Files/Self-matching/BT/{algo}_r={r:d}.csv'

  if os.path.exists(datapath):

    # Load data
    data = pd.read_csv(datapath)

    # Retrieve l_h
    l_h = np.unique(data.h)

    # --- Plots ------------------------------------------------------------

    # Accuracy
    ax[0].plot(data.h, data.g, '-', label=algo)
    
    # Structural quality
    ax[1].plot(data.h, data.q, '-', label=algo)
    
if l_h is not None:
  ax[0].plot(l_h, (l_h+1)/(2**(l_h+1) - 1), '--', color='w', label='Th')

ax[0].set_yscale('log')

ax[0].set_ylim([1e-3, 1])
ax[1].set_ylim([0, 1.01])

ax[0].set_xlabel('$h$')
ax[1].set_xlabel('$h$')

ax[0].set_ylabel('$\gamma$')
ax[1].set_ylabel('$q_s$')

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