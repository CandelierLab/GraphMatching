'''
Subgraph degradation
Edges with xi_m measurable attributes
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

directed = False
nA = 20
nRun = 500
# nRun = 1000

l_xi_m = np.arange(3)
# l_xi_m = np.arange(9)

force = True

# --------------------------------------------------------------------------

p_star = 2/nA
l_delta = np.linspace(0, 1, 11)
l_delta[-1] = 1-1/nA

# --------------------------------------------------------------------------

parser = argparse.ArgumentParser()

if not force:  
  parser.add_argument('-F', '--force', action='store_true')
  args = parser.parse_args()
  force = args.force

# ==========================================================================

ds = 'directed' if directed else 'undirected'

fname = project.root + f'/Files/Subgraph/ER/{ds}_edges_measurable_nA={nA:d}_nRun={nRun:d}.csv'

# Check existence
if os.path.exists(fname) and not force:
  sys.exit()

# Creating dataframe
df = pd.DataFrame(columns=['delta', 'zeta_m', 'g_Zager', 'g_GASM', 'q_Zager', 'q_GASM', 'g_Zager_std', 'g_GASM_std', 'q_Zager_std', 'q_GASM_std'])

k = 0

for xi_m in l_xi_m:

  print(f'--- Xi_m = {xi_m}') 

  for delta in l_delta:

    print(f'{nRun} iterations with delta={delta} ...', end='')
    start = time.time()
    
    g_Zager = []
    g_GASM = []
    q_Zager = []
    q_GASM = []

    for i in range(nRun):

      Ga = Gnp(nA, p_star, directed=directed)
      for i in range(xi_m):
        Ga.add_edge_attr('gauss')

      # Subgraph
      Gb, gt = Ga.degrade('vx_rm', delta=delta)

      # --- Zager

      if not xi_m:
        
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
    df.loc[k, 'delta'] = delta
    df.loc[k, 'xi_m'] = xi_m

    # Mean values
    if not xi_m:
      df.loc[k, 'g_Zager'] = np.mean(g_Zager)
      df.loc[k, 'q_Zager'] = np.mean(q_Zager)
    df.loc[k, 'g_GASM'] = np.mean(g_GASM)
    df.loc[k, 'q_GASM'] = np.mean(q_GASM)

    # Standard deviations
    if not xi_m:
      df.loc[k, 'g_Zager_std'] = np.std(g_Zager)
      df.loc[k, 'q_Zager_std'] = np.std(q_Zager)
    df.loc[k, 'g_GASM_std'] = np.std(g_GASM)
    df.loc[k, 'q_GASM_std'] = np.std(q_GASM)

    k += 1

    print('{:.02f} sec'.format((time.time() - start)))

# --- Save
    
print('Saving ...', end='')
start = time.time()

df.to_csv(fname)

print('{:.02f} sec'.format((time.time() - start)))