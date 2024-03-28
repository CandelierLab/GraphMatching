import os
import numpy as np
import pandas as pd
import time

import project
from Graph import *
from Comparison import *

os.system('clear')

# === Parameters ===========================================================

nA = 100
p_star = 2/nA
nRun = 1000

zeta = 1
l_nc = np.round(np.geomspace(1, 2*nA, 10)).astype(int)

force = True

# --------------------------------------------------------------------------

Nsub = np.linspace(0, nA, 11, dtype=int)
Nsub[0] = 1

# ==========================================================================

for nc in l_nc:

  print('--- Number of categories nc =', nc)

  fname = project.root + '/Files/Success ratios/nMeas_edges_nc/ER_nA={:d}_nc={:d}_nRun={:d}.csv'.format(nA, nc, nRun)

  # Skip if existing
  if os.path.exists(fname) and not force: continue

  # Creating dataframe
  gamma = pd.DataFrame()

  for n in Nsub:

    print('{:d} iterations with subgraph of size {:d} ...'.format(nRun, n), end='')
    start = time.time()
    
    g = np.empty(nRun)

    for i in range(nRun):

      Net = Network(nA)
      Net.set_rand_edges('ER', p_star)

      '''
      zeta = 1
      nc variable
      '''

      val = np.ceil(np.linspace(1e-10, 1, Net.nEd)*nc).astype('int')
      Net.add_edge_attr({'measurable': False, 'values': val})

      Sub, Idx = Net.subnet(n)

      M = matching(Net, Sub)

      # Correct matches
      g[i] = np.count_nonzero([Idx[m[1]]==m[0] for m in M])/n if n else 0

      gamma[n] = g

    print('{:.02f} sec'.format((time.time() - start)))

  # --- Save

  gamma.to_csv(fname)