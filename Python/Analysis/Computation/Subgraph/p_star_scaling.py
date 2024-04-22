'''
Searching for p*
'''

import os, sys
import argparse
import numpy as np
import time

import project
from Graph import *
from Comparison import *

# os.system('clear')

# === Parameters ===========================================================

directed = True

l_nA = [10, 20, 50, 100, 200, 500, 1000]

delta = 0.25

nRun = int(1e3)
# nRun = int(1e5)

nBin = int(1e4)

force = False

# --------------------------------------------------------------------------

parser = argparse.ArgumentParser()

if not force:  
  parser.add_argument('-F', '--force', action='store_true')
  args = parser.parse_args()
  force = args.force

# --------------------------------------------------------------------------

dname = project.root + '/Files/Subgraph/p_star/'

# ==========================================================================

for nA in l_nA:

  fname = dname + f'nA={nA:d}.txt'

  # Check existence
  if force or not os.path.exists(fname):

    print(f'Searching p_star, nA={nA}', end='', flush=True)
    start = time.time()

    # Definitions
    x = np.linspace(0, 4/nA, nBin)
    y = np.zeros(nBin)
    sigma = 2/nA

    # Preparation
    p = np.random.rand(nRun)*4/nA
    g = np.empty(nRun)

    for i in range(nRun):

      # Graphs
      Ga = Gnp(nA, p[i], directed=directed)
      Gb, gt = Ga.subgraph(delta=delta)

      # --- GASM

      C = Comparison(Ga, Gb)
      M = C.get_matching(algorithm='GASM')
      M.compute_accuracy(gt)

      g[i] = M.accuracy
      
      if g[i]>0:
        y -= np.log(g[i])*np.exp(-(((x-p[i])/sigma)**2)/2)/nRun

      if not i % (nRun/20):
        print('.', end='', flush=True)

    p_star = x[np.argmax(y)]

    print(f' p_star: {p_star}' + ' ({:.02f} sec)'.format((time.time() - start)))

    # --- Save

    with open(fname, 'w') as f:
      f.write(str(p_star))