import os
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

import project
from Graph import *
from Comparison import *

os.system('clear')

# === Parameters ===========================================================

n = 100
scale = 'lin'
# scale = 'log'

nRun = 100


# Degrees (average number of edges per node)
match scale:
  case 'lin':
    l_deg = np.linspace(0, n, 31)
  case 'log':
    l_deg = np.geomspace(1/n, n, 31)


# --------------------------------------------------------------------------

fname = project.root + f'/Files/Normalization/ER/{scale}_n={n:d}_nRun={nRun:d}.csv'

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

    NetB, Idx = NetA.shuffle()

    # --- Matching

    C = Comparison(NetA, NetB)
    M = C.get_matching(algorithm='GASM', normalization=1, info_avgScores=True)

    if 'avgX' in C.info and len(C.info['avgX'])>1:
      f[run] = C.info['avgX'][-1]/C.info['avgX'][-2]
    else:
      f[run] = 1

  fac[deg] = f

  print('{:.02f} sec'.format((time.time() - start)))

# === Save =================================================================

fac.to_csv(fname)