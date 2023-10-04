import os
import numpy as np
import pandas as pd
import time

import project
from Network import *
from Comparison import *

os.system('clear')

# === Parameters ===========================================================

nA = 500
p_star = 2/nA
nRun = 10000

# --------------------------------------------------------------------------

fname = project.root + '/Files/Success ratios/rho/ER_nA={:d}_nRun={:d}.csv'.format(nA, nRun)

Nsub = np.linspace(nA/10, nA, 10, dtype=int)

# ==========================================================================

# Creating dataframe
gamma = pd.DataFrame()

for n in Nsub:

  print('{:d} iterations with subgraph of size {:d} ...'.format(nRun, n), end='')
  start = time.time()
  
  g = np.empty(nRun)

  for i in range(nRun):

    Net = Network(nA)
    Net.set_rand_edges('ER', p_star)

    Sub, Idx = Net.subnet(n)

    M = matching(Net, Sub)

    # Correct matches
    g[i] = np.count_nonzero([Idx[m[1]]==m[0] for m in M])/n

    gamma[n] = g

  print('{:.02f} sec'.format((time.time() - start)))

# === Save =================================================================

gamma.to_csv(fname)