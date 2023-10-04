import os
import numpy as np
import pandas as pd
import time

import project
from Network import *
from Comparison import *

os.system('clear')

# === Parameters ===========================================================

nA = 100
p_star = 2/nA
nRun = 1000

l_xi = np.arange(9)

force = True

# --------------------------------------------------------------------------

Nsub = np.linspace(0, nA, 11, dtype=int)
Nsub[0] = 1

# ==========================================================================

for xi in l_xi:

  print(f'--- Xi = {xi}')

  fname = project.root + '/Files/Success ratios/Meas_edges/ER_nA={:d}_xi={:d}_nRun={:d}.csv'.format(nA, xi, nRun)

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

      for z in range(xi):
        Net.add_edge_attr('gauss')

      Sub, Idx = Net.subnet(n)

      M = matching(Net, Sub)

      # Correct matches
      g[i] = np.count_nonzero([Idx[m[1]]==m[0] for m in M])/n if n else 0

      gamma[n] = g

    print('{:.02f} sec'.format((time.time() - start)))

  # --- Save

  gamma.to_csv(fname)