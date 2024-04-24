'''
Circular ladder graph: average gamma and q
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

l_n = np.arange(1, 26)

nRun = 10000

force = True

# --------------------------------------------------------------------------

parser = argparse.ArgumentParser()

if not force:  
  parser.add_argument('-F', '--force', action='store_true')
  args = parser.parse_args()
  force = args.force

# ==========================================================================

fname = project.root + f'/Files/Self-matching/CL/{algo}_CL.csv'

# Check existence
if os.path.exists(fname) and not force:
  sys.exit()

pa.line(f'CL - {algo}')

# Creating dataframe
df = pd.DataFrame(columns=['n', 'eta', 'nRun', 'g', 'q', 'g_std', 'q_std'])

k = 0

for n in l_n:

  print(f'n={n:d}, {nRun:d} iterations ...', end='', flush=True)
  start = time.time()

  Ga = Graph(nx=nx.circular_ladder_graph(n))
  
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
  df.loc[k, 'n'] = n
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

