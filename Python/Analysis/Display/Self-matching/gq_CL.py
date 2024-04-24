'''
Circular ladder graph: average gamma and q
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

# --------------------------------------------------------------------------

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

  fname = project.root + f'/Files/Self-matching/CL/{algo}_CL.csv'

  if os.path.exists(fname):

    # Load data
    data = pd.read_csv(fname)

    # Retrieve l_n and l_eta

    l_n = np.unique(data.n)
    l_eta = np.unique(data.eta)

    # --- Accuracy

    for i, eta in enumerate(l_eta):

      # Accuracy
      ax[0].plot(data.n, data.g, '.', label=algo)

      # Structural quality
      ax[1].plot(data.n, data.q, '.-', label=algo)

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

#  --- Output --------------------------------------------------------------

if figfile is None:

  plt.show()

else:

  print(f'Saving file {figfile} ...', end='', flush=True)
  tref = time.time()

  plt.savefig(project.root + '/Figures/' + figfile)

  print(' {:.02f} sec'.format((time.time() - tref)))