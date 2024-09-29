'''
-- Display --

ER (Gnp): average gamma and q

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
nA = 20

# --------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename', help='File to save the figure')
args = parser.parse_args()
figfile = args.filename

# --------------------------------------------------------------------------

ds = 'directed' if directed else 'undirected'

# ==========================================================================

if figfile is None:
  os.system('clear')

# --- Display --------------------------------------------------------------

plt.style.use('dark_background')
fig, ax = plt.subplots(1, 2, figsize=(20,10))

for algo in l_algo:

  datapath = project.root + f'/Files/Self-matching/ER/{algo}_{ds}_nA={nA:d}.csv'

  if os.path.exists(datapath):

    # Load data
    data = pd.read_csv(datapath)

    # Retrieve l_p

    l_p = np.unique(data.p)

    # Accuracy
    ax[0].plot(data.p, data.g, '-', label=algo)

    # Structural quality
    ax[1].plot(data.p, data.q, '-', label=algo)

# ax[1].set_yscale('log')

ax[0].set_ylim([0, 1])
ax[1].set_ylim([0, 1.001])

ax[0].set_xlabel('p')
ax[1].set_xlabel('p')

ax[0].set_ylabel('$\gamma$')
ax[1].set_ylabel('$q$')

ax[0].legend()
ax[1].legend()

# ax[0].set_xscale('log')
ax[0].grid(True)

#  --- Output --------------------------------------------------------------

if figfile is None:

  plt.show()

else:

  print(f'Saving file {figfile} ...', end='', flush=True)
  tref = time.time()

  plt.savefig(project.root + '/Figures/' + figfile)

  print(' {:.02f} sec'.format((time.time() - tref)))