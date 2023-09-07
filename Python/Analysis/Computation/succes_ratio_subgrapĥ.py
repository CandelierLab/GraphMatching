import os
import numpy as np
import pandas as pd
import time

import project
from Network import *
from Comparison import *

os.system('clear')

# === Parameters ===========================================================

nA = 15
Nsub = list(range(1, nA+1))

p = 0.5

power = 0.5

nRun = 1000

# --------------------------------------------------------------------------

fname = project.root + '/Files/Success ratios/ER_p={:.02f}_nA={:d}_nRun={:d}_power={:.01f}.csv'.format(p, nA, nRun, power)

# ==========================================================================

# Creating dataframe
gamma = pd.DataFrame()

for n in Nsub:

  print('{:d} iterations with subgraph of size {:d} ...'.format(nRun, n), end='')
  start = time.time()
  
  g = np.empty(nRun)

  for i in range(nRun):

    Net = Network(nA)
    Net.set_rand_edges('ER', p)

    Sub, Idx = Net.subnet(n)    

    if not Sub.Adj.any():
      g[i] = 0

    else:

      M = matching(Net, Sub, power=power)

      # Correct matches
      g[i] = np.count_nonzero([Idx[m[1]]==m[0] for m in M])/n

    gamma[n] = g

  print('{:.02f} sec'.format((time.time() - start)))

# === Save =================================================================

gamma.to_csv(fname)