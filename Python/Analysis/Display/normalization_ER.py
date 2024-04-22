import os
import argparse
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

import project

# === Parameters ===========================================================

nA = 100
scale = 'lin'
# scale = 'log'
nRun = 100

# --------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename', help='File to save the figure')
args = parser.parse_args()
figfile = args.filename

# --------------------------------------------------------------------------

datapath = project.root + f'/Files/Normalization/ER/{scale}_n={nA:d}_nRun={nRun:d}.csv'

# ==========================================================================

if figfile is None:
  os.system('clear')

if os.path.exists(datapath):

  # Load data
  F = pd.read_csv(datapath, index_col=0)

  # x-values
  x = np.array([float(i) for i in list(F)])

  # Compute mean and std
  mv = F.mean()
  s = F.std()

# Simple normalization factor
f0 = np.minimum(4*x**2 + 1, 4*(nA-x)**2 + 1)

# === Display =================================================================

plt.style.use('dark_background')

fig, ax = plt.subplots()

ax.plot(x, f0, '--', linewidth=1)
ax.fill_between(x, mv-s, mv+s,
    alpha=0.5, facecolor='#dddddd')

ax.plot(x, mv, '.')

# ax.plot(x, f1, 'r-', linewidth=1)

ax.set_xlabel('Average degree')
ax.set_ylabel('Normalization factor')

if scale=='log':
  ax.set_xscale('log')
ax.set_yscale('log')

ax.set_title('n = {:d}'.format(nA))

# ax.set_ylim(1, 6e4)

#  --- Output --------------------------------------------------------------

if figfile is None:

  plt.show()

else:

  print(f'Saving file {figfile} ...', end='', flush=True)
  tref = time.time()

  plt.savefig(project.root + '/Figures/' + figfile)

  print(' {:.02f} sec'.format((time.time() - tref)))