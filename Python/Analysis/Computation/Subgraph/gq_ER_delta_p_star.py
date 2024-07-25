'''
Subgraph degradation
Erdo-Renyi: average gamma and q as a function of delta, nA for p=p_star
'''

import os, sys
import argparse
import numpy as np
import pandas as pd
import time
from alive_progress import alive_bar

import project
from Graph import *
from Comparison import *

os.system('clear')

# === Parameters ===========================================================

directed = True
# l_nA = [10, 20, 50, 100, 200, 500, 1000]
l_nA = [50]

nRun = 10000
# nRun = 100

force = True

# --------------------------------------------------------------------------

parser = argparse.ArgumentParser()

if not force:  
  parser.add_argument('-F', '--force', action='store_true')
  args = parser.parse_args()
  force = args.force
  
# --------------------------------------------------------------------------

dname = project.root + '/Files/Subgraph/delta/'

# ==========================================================================

for nA in l_nA:

  p_star = 2/nA

  # l_delta = np.linspace(0, 1, 11)
  # l_delta[10] = 1-1/nA
  # l_delta = np.unique(l_delta)

  l_delta = np.geomspace(max(0.01,1/nA), 1-1/nA, 11)
  
  # --------------------------------------------------------------------------

  fname = dname + f'ER_nA={nA}_nRun={nRun}.csv'

  # Skip if already existing
  if os.path.exists(fname) and not force: continue

  # Creating dataframe
  gamma = pd.DataFrame()

  for delta in l_delta:

    g = np.empty(nRun)

    with alive_bar(nRun) as bar:

      bar.title = f'delta={delta}'

      for i in range(nRun):

        # Graphs
        Ga = Gnp(nA, p_star, directed=directed)
        Gb, gt = Ga.subgraph(delta=delta)

        # --- GASM

        C = Comparison(Ga, Gb)
        M = C.get_matching(algorithm='GASM')
        M.compute_accuracy(gt)

        g[i] = M.accuracy

        bar()

      gamma[delta] = g

    # === Save =================================================================

    gamma.to_csv(fname)