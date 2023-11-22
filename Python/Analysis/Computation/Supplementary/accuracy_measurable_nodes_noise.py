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

zeta = 1
l_sigma = [0, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5]

force = False

# --------------------------------------------------------------------------

Nsub = np.linspace(0, nA, 11, dtype=int)
Nsub[0] = 1

# ==========================================================================

for sigma in l_sigma:

  print(f'--- Sigma = {sigma}')

  fname = project.root + '/Files/Success ratios/Meas_nodes_noise/ER_nA={:d}_sigma={:f}_nRun={:d}.csv'.format(nA, sigma, nRun)

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
        Net.add_node_attr('gauss')

      Sub, Idx = Net.subnet(n)

      # Add noise
      Sub.node_attr[0]['values'] += sigma*np.random.randn(Sub.node_attr[0]['values'].size)

      M = matching(Net, Sub)

      # Correct matches
      g[i] = np.count_nonzero([Idx[m[1]]==m[0] for m in M])/n if n else 0

      gamma[n] = g

    print('{:.02f} sec'.format((time.time() - start)))

  # --- Save

  gamma.to_csv(fname)