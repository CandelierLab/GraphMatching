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

import paprint as pa

os.system('clear')

# === Parameters ===========================================================

# algo = 'FAQ'
algo = '2opt'
# algo = 'Zager'
# algo = 'GASM'

r = 2
l_h = np.arange(2,11)

# r = 3
# l_h = np.arange(2,8)

l_eta = [1e-10] #np.logspace(-6, -14, 5)

force = False

# --------------------------------------------------------------------------

parser = argparse.ArgumentParser()

if not force:  
  parser.add_argument('-F', '--force', action='store_true')
  args = parser.parse_args()
  force = args.force

# ==========================================================================

fname = project.root + f'/Files/Self-matching/BT/{algo}_r={r:d}.csv'

# Check existence
if os.path.exists(fname) and not force:
  sys.exit()

pa.line(f'BT - {algo}')

# Creating dataframe
# df = pd.DataFrame(columns=['h', 'eta', 'nRun', 'g_FAQ', 'g_2opt', 'g_Zager', 'g_GASM', 'q_FAQ', 'g_2opt', 'q_Zager', 'q_GASM', 'g_FAQ_std', 'g_2opt_std', 'g_Zager_std', 'g_GASM_std', 'q_FAQ_std', 'g_2opt_std',  'q_Zager_std', 'q_GASM_std'])

df = pd.DataFrame(columns=['h', 'eta', 'nRun', 'g', 'q', 'g_std', 'q_std'])

k = 0

for h in l_h:

  print(f'h={h:d}')

  Ga = Graph(nx=nx.balanced_tree(r, h))

  # Number of runs
  nRun = int(np.ceil(2.0**(13-h)))

  for eta in l_eta:

    print(f'{nRun:d} iterations: eta={eta:.05f} ...', end='')
    start = time.time()
    
    g = []
    q = []

    for i in range(nRun):

      Gb, gt = Ga.shuffle()

      # --- FAQ

      C = Comparison(Ga, Gb)
      M = C.get_matching(algorithm=algo)
      M.compute_accuracy(gt)

      g.append(M.accuracy)
      q.append(M.structural_quality)

    # --- Store
      
    # Parameters
    df.loc[k, 'h'] = h
    df.loc[k, 'eta'] = eta
    df.loc[k, 'nRun'] = nRun

    # Mean values
    df.loc[k, 'g'] = np.mean(g)
    df.loc[k, 'q'] = np.mean(q)

    # Standard deviations
    df.loc[k, 'g_std'] = np.std(g)
    df.loc[k, 'q_std'] = np.std(q)

    k += 1

    print('{:.02f} sec'.format((time.time() - start)))

# --- Save
    
print('Saving ...', end='')
start = time.time()

df.to_csv(fname)

print('{:.02f} sec'.format((time.time() - start)))

