'''
Compute normalization factors
'''

import os, sys
import argparse
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

import project
from Graph import *
from Comparison import *

os.system('clear')

# === Parameters ===========================================================

nA = 100
scale = 'lin'
# scale = 'log'

nRun = 100


# Degrees (average number of edges per node)
match scale:
  case 'lin':
    l_deg = np.linspace(0, nA, 31)
  case 'log':
    l_deg = np.geomspace(1/nA, nA, 31)

l_deg = [51]

force = True

# --------------------------------------------------------------------------

parser = argparse.ArgumentParser()

if not force:  
  parser.add_argument('-F', '--force', action='store_true')
  args = parser.parse_args()
  force = args.force

# ==========================================================================


fname = project.root + f'/Files/Normalization/ER/{scale}_n={nA:d}_nRun={nRun:d}.csv'

# Check existence
if os.path.exists(fname) and not force:
  sys.exit()

fac = pd.DataFrame()

for deg in l_deg:

  print('deg {:.01f} ...'.format(deg), end='')
  start = time.time()

  f = np.empty(nRun)

  for run in range(nRun):

    # --- Network

    Ga = Gnm(nA, deg*nA)
    Gb, gt = Ga.shuffle()

    # --- Matching

    C = Comparison(Ga, Gb)
    M = C.get_matching(algorithm='GASM', normalization=1, info_avgScores=True)

    if 'avgX' in C.info and len(C.info['avgX'])>1:
      f[run] = C.info['avgX'][-1]/C.info['avgX'][-2]
    else:
      f[run] = 1

  fac[deg] = f

  print('{:.02f} sec'.format((time.time() - start)))

# === Save =================================================================

fac.to_csv(fname)