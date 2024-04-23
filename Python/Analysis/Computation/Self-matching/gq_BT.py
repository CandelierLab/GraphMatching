'''
Balanced tree: average gamma and q
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

r = 2
l_h = np.arange(2,11)

# r = 3
# l_h = np.arange(2,8)

l_eta = [1e-10] #np.logspace(-6, -14, 5)

force = True

# --------------------------------------------------------------------------

parser = argparse.ArgumentParser()

if not force:  
  parser.add_argument('-F', '--force', action='store_true')
  args = parser.parse_args()
  force = args.force

# ==========================================================================

fname = project.root + f'/Files/Self-matching/BT/r={r:d}.csv'

# Check existence
if os.path.exists(fname) and not force:
  sys.exit()

# Creating dataframe
df = pd.DataFrame(columns=['h', 'eta', 'nRun', 'g_FAQ', 'g_2opt', 'g_Zager', 'g_GASM', 'q_FAQ', 'g_2opt', 'q_Zager', 'q_GASM', 'g_FAQ_std', 'g_2opt_std', 'g_Zager_std', 'g_GASM_std', 'q_FAQ_std', 'g_2opt_std',  'q_Zager_std', 'q_GASM_std'])

k = 0

for h in l_h:

  print(f'h={h:d}')

  Ga = Graph(nx=nx.balanced_tree(r, h))

  # Number of runs
  nRun = int(np.ceil(2.0**(13-h)))

  for eta in l_eta:

    print(f'{nRun:d} iterations: eta={eta:.05f} ...', end='')
    start = time.time()
    
    g_FAQ = []
    q_FAQ = []
    g_2opt = []
    q_2opt = []
    g_Zager = []
    q_Zager = []
    g_GASM = []
    q_GASM = []

    for i in range(nRun):

      Gb, gt = Ga.shuffle()

      # --- FAQ

      C = Comparison(Ga, Gb)
      M = C.get_matching(algorithm='FAQ')
      M.compute_accuracy(gt)

      g_FAQ.append(M.accuracy)
      q_FAQ.append(M.structural_quality)

      # --- 2opt

      C = Comparison(Ga, Gb)
      M = C.get_matching(algorithm='2opt')
      M.compute_accuracy(gt)

      g_2opt.append(M.accuracy)
      q_2opt.append(M.structural_quality)

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
    df.loc[k, 'h'] = h
    df.loc[k, 'eta'] = eta
    df.loc[k, 'nRun'] = nRun

    # Mean values
    df.loc[k, 'g_FAQ'] = np.mean(g_FAQ)
    df.loc[k, 'q_FAQ'] = np.mean(q_FAQ)
    df.loc[k, 'g_2opt'] = np.mean(g_2opt)
    df.loc[k, 'q_2opt'] = np.mean(q_2opt)
    df.loc[k, 'g_Zager'] = np.mean(g_Zager)
    df.loc[k, 'q_Zager'] = np.mean(q_Zager)
    df.loc[k, 'g_GASM'] = np.mean(g_GASM)
    df.loc[k, 'q_GASM'] = np.mean(q_GASM)

    # Standard deviations
    df.loc[k, 'g_FAQ_std'] = np.std(g_FAQ)
    df.loc[k, 'q_FAQ_std'] = np.std(q_FAQ)
    df.loc[k, 'g_2opt_std'] = np.std(g_2opt)
    df.loc[k, 'q_2opt_std'] = np.std(q_2opt)
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

