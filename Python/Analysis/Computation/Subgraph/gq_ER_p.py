'''
Subgraph degradation
Erdo-Renyi: average gamma and q as a function of p
'''

import os, sys
import argparse
import numpy as np
import pandas as pd
import time

import project
from Graph import *
from Comparison import *

os.system('clear')

# === Parameters ===========================================================

nA = 20
l_directed = [False, True]
l_p = np.linspace(0, 1, 101)
l_delta = np.linspace(0, 0.9, 10)
nRun = 10000
force = False

# --------------------------------------------------------------------------

parser = argparse.ArgumentParser()

if not force:  
  parser.add_argument('-F', '--force', action='store_true')
  args = parser.parse_args()
  force = args.force

# --------------------------------------------------------------------------

# Nsub = list(range(2, nA+1, 2))
Nsub = np.linspace(nA/10, nA, 10, dtype=int)

# ==========================================================================

fname = project.root + f'/Files/Subgraph/ER/ER_nA={nA:d}_nRun={nRun:d}.csv'

# Check existence
if os.path.exists(fname) and not force:
  sys.exit()

# Creating dataframe
df = pd.DataFrame(columns=['directed', 'p', 'delta', 'g_Zager', 'g_GASM', 'q_Zager', 'q_GASM', 'g_Zager_std', 'g_GASM_std', 'q_Zager_std', 'q_GASM_std'])

k = 0

for p in l_p:

  print(f'p={p:0.2f}')

  for delta in l_delta:

    print(f'delta={delta:.2f} ({nRun:d} iterations, x2) ...', end='', flush=True)

    for d in l_directed:

      start = time.time()
      
      g_Zager = []
      g_GASM = []
      q_Zager = []
      q_GASM = []

      for i in range(nRun):

        # Graphs
        Ga = Gnp(nA, p, directed=d)
        Gb, gt = Ga.subgraph(delta=delta)

        # --- Zager

        C = Comparison(Ga, Gb)
        M = C.get_matching(algorithm='Zager')
        M.compute_accuracy(gt)

        g_Zager.append(M.accuracy)
        q_Zager.append(M.structural_quality)

        # --- GASM

        C = Comparison(Ga, Gb)
        M = C.get_matching(algorithm='GASM')
        M.compute_accuracy(gt)

        g_GASM.append(M.accuracy)
        q_GASM.append(M.structural_quality)

      # --- Store
        
      # Parameters
      df.loc[k, 'directed'] = d
      df.loc[k, 'delta'] = delta
      df.loc[k, 'p'] = p

      # Mean values
      df.loc[k, 'g_Zager'] = np.mean(g_Zager)
      df.loc[k, 'q_Zager'] = np.mean(q_Zager)
      df.loc[k, 'g_GASM'] = np.mean(g_GASM)
      df.loc[k, 'q_GASM'] = np.mean(q_GASM)

      # Standard deviations
      df.loc[k, 'g_Zager_std'] = np.std(g_Zager)
      df.loc[k, 'q_Zager_std'] = np.std(q_Zager)
      df.loc[k, 'g_GASM_std'] = np.std(g_GASM)
      df.loc[k, 'q_GASM_std'] = np.std(q_GASM)

      k += 1

    print('{:.02f} sec'.format((time.time() - start)))

# --- Save
    
print('Saving ...', end='', flush=True)
start = time.time()

df.to_csv(fname)

print('{:.02f} sec'.format((time.time() - start)))
