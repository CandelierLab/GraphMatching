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
# algo = '2opt'
# algo = 'Zager'
algo = 'GASM'

r = 2
l_h = np.arange(2,11)

force = True

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
df = pd.DataFrame(columns=['h', 'eta', 'nRun', 'g', 'q', 'g_std', 'q_std'])

k = 0

for h in l_h:

  # Number of runs
  match algo:
    case '2opt':

      '''
      Running times for 2opt:
      h=2, 512 iterations ... 0.32 sec    0.0006 s/run
      h=3, 256 iterations ... 0.84 sec    0.003 s/run
      h=4, 128 iterations ... 3.51 sec    0.03  s/run
      h=5, 64 iterations ... 20.93 sec    0.33 s/run
      h=6, 32 iterations ... 176.74 sec   5,5 s/run
      h=7, 16 iterations ... 2212.20 sec  138 s/run
      h=8, 1 iterations ... 4774.68 sec   4775 s/run
      '''

      if h<8:
        nRun = int(np.ceil(2.0**(11-h)))
      elif h==8:
        nRun = 1
      else:
        continue

    case _:
      nRun = int(np.ceil(2.0**(13-h)))
      
  print(f'h={h:d}, {nRun:d} iterations ...', end='', flush=True)
  start = time.time()

  # Reference graph
  Ga = Graph(nx=nx.balanced_tree(r, h))

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
  df.loc[k, 'nRun'] = nRun

  # Mean values
  df.loc[k, 'g'] = np.mean(g)
  df.loc[k, 'q'] = np.mean(q)

  # Standard deviations
  df.loc[k, 'g_std'] = np.std(g)
  df.loc[k, 'q_std'] = np.std(q)

  k += 1

  print(' {:.02f} sec'.format((time.time() - start)))

# --- Save
    
print('Saving ...', end='')
start = time.time()

df.to_csv(fname)

print('{:.02f} sec'.format((time.time() - start)))

