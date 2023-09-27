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

n = 100

# Degrees (average number of edges per node)
l_deg = np.geomspace(1/n, n, 29)

nRun = 100

# --------------------------------------------------------------------------

fname = project.root + '/Files/Normalization/ER/n={:d}_nRun={:d}.csv'.format(n, nRun)

# === Functions ============================================================

def probe(V, param, out):

  f = np.mean(V['X'])

  # Output
  out.append(f)

# ==========================================================================

fac = pd.DataFrame()

for deg in l_deg:

  print('deg {:.01f} ...'.format(deg), end='')
  start = time.time()

  f = np.empty(nRun)

  for run in range(nRun):

    # --- Network

    NetA = Network(n)
    NetA.set_rand_edges('ER', int(deg*n))

    NetB, Icorr = NetA.shuffle()

    # NetB = Network(int(n))
    # NetB.set_rand_edges('ER', int(deg*n))

    # --- Convergence

    # Scores
    X, Y, output = scores(NetA, NetB, normalization=1,
                          i_function=probe, initial_evaluation=True)

    f[run] = output[-1]/output[-2]

  fac[deg] = f

  print('{:.02f} sec'.format((time.time() - start)))

# === Save =================================================================

fac.to_csv(fname)