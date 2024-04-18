'''
Erdo-Renyi: average gamma and q
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

directed = True
nA = 20
l_p = np.linspace(0,1,101)
# l_eta = np.logspace(-14, -2, 7)
l_eta = [1e-10]
nRun = 10000

force = False

# --------------------------------------------------------------------------

parser = argparse.ArgumentParser()

if not force:  
  parser.add_argument('-F', '--force', action='store_true')
  args = parser.parse_args()
  force = args.force

# ==========================================================================

ds = 'directed' if directed else 'undirected'

fname = project.root + f'/Files/Self-matching/ER/{ds}_nA={nA:d}_nRun={nRun:d}.csv'

# Check existence
if os.path.exists(fname) and not force:
  sys.exit()
  
# Creating dataframe
df = pd.DataFrame(columns=['p', 'eta', 'g_FAQ', 'g_Zager', 'g_GASM', 'q_FAQ',  'q_Zager', 'q_GASM', 'g_FAQ_std', 'g_Zager_std', 'g_GASM_std', 'q_FAQ_std',  'q_Zager_std', 'q_GASM_std'])

k = 0

for p in l_p:

  print(f'{nRun:d} iterations: p={p:.02f} ...', end='', flush=True)
  start = time.time()

  for eta in l_eta:
    
    g_FAQ = []
    q_FAQ = []
    g_Zager = []
    g_GASM = []
    q_Zager = []
    q_GASM = []

    for i in range(nRun):

      Ga = Gnp(nA, p, directed=directed)
      Gb, gt = Ga.shuffle()

      # --- FAQ

      C = Comparison(Ga, Gb)
      M = C.get_matching(algorithm='FAQ')
      M.compute_accuracy(gt)

      g_FAQ.append(M.accuracy)
      q_FAQ.append(M.structural_quality)

      # --- Zager

      C = Comparison(Ga, Gb)
      M = C.get_matching(algorithm='Zager')
      M.compute_accuracy(gt)

      g_Zager.append(M.accuracy)
      q_Zager.append(M.structural_quality)

      # --- GASM

      C = Comparison(Ga, Gb)
      M = C.get_matching(algorithm='GASM', eta=eta)
      M.compute_accuracy(gt)

      g_GASM.append(M.accuracy)
      q_GASM.append(M.structural_quality)

    # --- Store
      
    # Parameters
    df.loc[k, 'p'] = p
    df.loc[k, 'eta'] = eta

    # Mean values
    df.loc[k, 'g_FAQ'] = np.mean(g_FAQ)
    df.loc[k, 'q_FAQ'] = np.mean(q_FAQ)
    df.loc[k, 'g_Zager'] = np.mean(g_Zager)
    df.loc[k, 'q_Zager'] = np.mean(q_Zager)
    df.loc[k, 'g_GASM'] = np.mean(g_GASM)
    df.loc[k, 'q_GASM'] = np.mean(q_GASM)

    # Standard deviations
    df.loc[k, 'g_FAQ_std'] = np.std(g_FAQ)
    df.loc[k, 'q_FAQ_std'] = np.std(q_FAQ)
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

