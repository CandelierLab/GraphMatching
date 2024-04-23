'''
Star-branched graph: average gamma and q
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

l_k = np.arange(2,11)
l_n = np.arange(1,11)

l_eta = [1e-3]
nRun = 1000

force = False

# --------------------------------------------------------------------------

parser = argparse.ArgumentParser()

if not force:  
  parser.add_argument('-F', '--force', action='store_true')
  args = parser.parse_args()
  force = args.force

# ==========================================================================

fname = project.root + f'/Files/Self-matching/SB/nRun={nRun:d}.csv'

# Check existence
if os.path.exists(fname) and not force:
  sys.exit()
  
# Creating dataframe
df = pd.DataFrame(columns=['k', 'n', 'eta', 'nRun', 'g_FAQ', 'g_2opt', 'g_Zager', 'g_GASM', 'q_FAQ', 'g_2opt', 'q_Zager', 'q_GASM', 'g_FAQ_std', 'g_2opt_std', 'g_Zager_std', 'g_GASM_std', 'q_FAQ_std', 'g_2opt_std',  'q_Zager_std', 'q_GASM_std'])


i = 0

for k in l_k:
  for n in l_n:

    print(f'k={k:d}, n={n:d}, ', end='')

    Ga = star_branched(k,n, directed=True)

    for eta in l_eta:

      print(f'{nRun:d} iterations: eta={eta:.05f} ...', end='')
      start = time.time()
      
      g_FAQ = []
      q_FAQ = []
      g_2opt = []
      q_2opt = []
      g_Zager = []
      g_GASM = []
      q_Zager = []
      q_GASM = []

      for j in range(nRun):

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
      df.loc[i, 'k'] = k
      df.loc[i, 'n'] = n
      df.loc[i, 'eta'] = eta
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

      i += 1

      print('{:.02f} sec'.format((time.time() - start)))

# --- Save
    
print('Saving ...', end='')
start = time.time()

df.to_csv(fname)

print('{:.02f} sec'.format((time.time() - start)))

