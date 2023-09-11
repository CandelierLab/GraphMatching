import os
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

import project
from Network import *
from Comparison import *

os.system('clear')

# === Parameters ===========================================================

n = 10

# Average number of edges per node
l_nepn = np.geomspace(1/n, n, 29)

nIter = 10

nRun = 100

# --------------------------------------------------------------------------

fname = project.root + '/Files/Normalization/ER/n={:d}_nIter={:d}_nRun={:d}.csv'.format(n, nIter, nRun)

# === Functions ============================================================

def probe(V, param, out):

  f = np.mean(V['X'])

  # Output
  out.append(f)

# ==========================================================================

fac = pd.DataFrame()

for nepn in l_nepn:

  print('Nepn {:.01f} ...'.format(nepn), end='')
  start = time.time()

  f = np.empty(nRun)

  for run in range(nRun):

    # --- Network

    NetA = Network(n)
    NetA.set_rand_edges('ER', int(nepn*n))

    NetB, Icorr = NetA.shuffle()

    # NetB = Network(int(n))
    # NetB.set_rand_edges('ER', int(nepn*n))

    # --- Convergence

    # Scores
    X, Y, output = scores(NetA, NetB, nIter=nIter, normalization=1,
                          i_function=probe, initial_evaluation=True)

    f[run] = output[-1]/output[-2]

  fac[nepn] = f

  print('{:.02f} sec'.format((time.time() - start)))

# === Save =================================================================

fac.to_csv(fname)