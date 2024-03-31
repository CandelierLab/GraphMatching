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

l_zeta = [1] #np.arange(9)

force = True

# --------------------------------------------------------------------------

nSub = np.linspace(0, nA, 11, dtype=int)
nSub[0] = 1

# ==========================================================================

for zeta in l_zeta:

  print(f'--- Zeta = {zeta}')

  fname = project.root + f'/Files/Gbgraph/ER/Meas_nodes_nA={nA:d}_zeta={zeta:d}_nRun={nRun:d}.csv'

  # Skip if existing
  if os.path.exists(fname) and not force: continue

  # Creating dataframe
  gamma = pd.DataFrame()

  for n in nSub:

    print('{:d} iterations with Gbgraph of size {:d} ...'.format(nRun, n), end='')
    start = time.time()
    
    g = np.empty(nRun)

    for i in range(nRun):

      Ga = Gnp(nA, p_star)

      for z in range(zeta):
        Ga.add_node_attr('gauss')

      Gb, Idx = Ga.subgraph(n)

      C = Comparison(Ga, Gb)
      M = C.get_matching(algorithm='GASM')
      M.compute_accuracy(Idx)

      gamma[n] = M.accuracy

    print('{:.02f} sec'.format((time.time() - start)))

  # --- Save

  gamma.to_csv(fname)