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

l_zeta = np.arange(11)
nc = 2

force = False

# --------------------------------------------------------------------------

Nsub = np.linspace(0, nA, 11, dtype=int)
Nsub[0] = 1

# ==========================================================================

for zeta in l_zeta:

  print(f'--- Zeta = {zeta}')

  fname = project.root + '/Files/Success ratios/nMeas_nodes_zeta/ER_nA={:d}_zeta={:d}_nRun={:d}.csv'.format(nA, zeta, nRun)

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

      for z in range(zeta):
        
        '''
        zeta variable
        nc = 2
        '''

        val = np.floor(np.linspace(0, 1, Net.nNd)*nc).astype('int')
        np.random.shuffle(val)
        Net.add_node_attr({'measurable': False, 'values': val})

      Sub, Idx = Net.subnet(n)

      M = matching(Net, Sub)

      # Correct matches
      g[i] = np.count_nonzero([Idx[m[1]]==m[0] for m in M])/n if n else 0

      gamma[n] = g

    print('{:.02f} sec'.format((time.time() - start)))

  # --- Save

  gamma.to_csv(fname)