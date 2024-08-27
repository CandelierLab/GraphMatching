import os
import argparse
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

import project

# === Parameters ===========================================================

l_directed = [True, False]
l_nA = [10, 20, 50]

# --------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename', help='File to save the figure')
args = parser.parse_args()
figfile = args.filename

# --------------------------------------------------------------------------

# ==========================================================================

if figfile is None:
  os.system('clear')

plt.style.use('dark_background')
fig, ax = plt.subplots()

for directed in l_directed:

  ds = 'directed' if directed else 'undirected'

  for nA in l_nA:

    fname = project.root + f'/Files/Diameter/{ds}_nA={nA:d}.csv'

    if os.path.exists(fname):

      # Load data
      F = pd.read_csv(fname, index_col=0)

      # Plot
      if directed:
        ax.plot(F.p, F.diameter, '.-', label=nA)
      else:
        ax.plot(F.p, F.diameter, '--')

ax.legend()

ax.set_xlabel('$p$')
ax.set_ylabel(r'$\Delta$')

#  --- Output --------------------------------------------------------------

if figfile is None:

  plt.show()

else:

  print(f'Saving file {figfile} ...', end='', flush=True)
  tref = time.time()

  plt.savefig(project.root + '/Figures/' + figfile)

  print(' {:.02f} sec'.format((time.time() - tref)))
