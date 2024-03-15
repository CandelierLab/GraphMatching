import os
import numpy as np
import pandas as pd
import time

import project
from Network import *
from Comparison import *

os.system('clear')

# === Parameters ===========================================================

nA = 20
l_p = np.linspace(0,1,101)
nRun = 1000

# --------------------------------------------------------------------------

# Nsub = list(range(2, nA+1, 2))
Nsub = np.linspace(nA/10, nA, 10, dtype=int)

# ==========================================================================

for p in l_p:

  print(f'p={p:0.2f}')

  fname = project.root + '/Files/Success ratios/p/ER_p={:.02f}_nA={:d}_nRun={:d}.csv'.format(p, nA, nRun)

  # Creating dataframe
  gamma = pd.DataFrame()

  for n in Nsub:

    print('{:d} iterations: rho={:.2f} ...'.format(nRun, n/nA), end='')
    start = time.time()
    
    g = np.empty(nRun)

    for i in range(nRun):

      Net = Network(nA)
      Net.set_rand_edges('ER', p)

      Sub, Idx = Net.subnet(n)

      M = matching(Net, Sub)

      # Correct matches
      g[i] = np.count_nonzero([Idx[m[1]]==m[0] for m in M])/n

    gamma[n] = g

    print('{:.02f} sec'.format((time.time() - start)))

  # --- Save

  gamma.to_csv(fname)
