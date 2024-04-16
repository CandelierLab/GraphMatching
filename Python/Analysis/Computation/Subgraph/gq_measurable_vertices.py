'''
Subgraph degradation
Vertices with zeta measurable attributes
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

nA = 15
p = 0.2
nRun = 500

l_zeta = [0, 1] #np.arange(9)

force = True

# --------------------------------------------------------------------------

l_delta = np.arange(nA)/nA

# --------------------------------------------------------------------------

parser = argparse.ArgumentParser()

if not force:  
  parser.add_argument('-F', '--force', action='store_true')
  args = parser.parse_args()
  force = args.force

# ==========================================================================

fname = project.root + f'/Files/Subgraph/ER/Meas_vertices_nA={nA:d}_nRun={nRun:d}.csv'

# Check existence
if os.path.exists(fname) and not force:
  sys.exit()

# Creating dataframe
df = pd.DataFrame(columns=['delta', 'zeta', 'g_Zager', 'g_GASM', 'q_Zager', 'q_GASM', 'g_Zager_std', 'g_GASM_std', 'q_Zager_std', 'q_GASM_std'])

k = 0

for zeta in l_zeta:

  print(f'--- Zeta = {zeta}')

  for delta in l_delta:

    print(f'{nRun} iterations with delta={delta} ...', end='')
    start = time.time()
    
    g_Zager = []
    g_GASM = []
    q_Zager = []
    q_GASM = []

    for i in range(nRun):

      Ga = Gnp(nA, p, directed=zeta)
      # for i in range(zeta):
      #   Ga.add_vrtx_attr('gauss')

      # Subgraph
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
    df.loc[k, 'delta'] = delta
    df.loc[k, 'zeta'] = zeta

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
    
print('Saving ...', end='')
start = time.time()

df.to_csv(fname)

print('{:.02f} sec'.format((time.time() - start)))