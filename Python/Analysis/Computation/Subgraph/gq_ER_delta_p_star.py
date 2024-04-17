import os
import numpy as np
import pandas as pd
import time

import project
from Graph import *
from Comparison import *

os.system('clear')

# === Parameters ===========================================================

directed = True
l_nA = [10, 20, 50, 100, 200, 500, 1000]
l_delta = np.linspace(0, 1, 11)

# nRun = 1e4
nRun = 100

force = True

# --------------------------------------------------------------------------

dname = project.root + '/Files/Subgraph/delta/'

# ==========================================================================

for nA in l_nA:

  p_star = 2/nA
  
  # --------------------------------------------------------------------------

  fname = dname + f'ER_nA={nA:d}_nRun={nRun:d}.csv'

  # Skip if already existing
  if os.path.exists(fname) and not force: continue

  # Creating dataframe
  gamma = pd.DataFrame()

  for delta in l_delta:

    print(f'nA={nA}, delta={delta} ...', end='', flush=True)
    start = time.time()
    
    g = np.empty(nRun)

    for i in range(nRun):

      # Graphs
      Ga = Gnp(nA, p_star, directed=directed)
      Gb, gt = Ga.subgraph(delta=delta)

      # --- GASM

      C = Comparison(Ga, Gb)
      M = C.get_matching(algorithm='GASM')
      M.compute_accuracy(gt)

      g[i] = M.accuracy

    gamma[delta] = g

    print('{:.02f} sec'.format((time.time() - start)))

  # === Save =================================================================

  gamma.to_csv(fname)