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

import paprint as pa

os.system('clear')

# === Parameters ===========================================================

# algo = 'FAQ'
# algo = '2opt'
algo = 'Zager'
# algo = 'GASM'

directed = False
nA = 20
l_p = np.linspace(0,1,101)

nRun = 10000

force = True

# --------------------------------------------------------------------------

parser = argparse.ArgumentParser()

if not force:  
  parser.add_argument('-F', '--force', action='store_true')
  args = parser.parse_args()
  force = args.force

# ==========================================================================

ds = 'directed' if directed else 'undirected'

fname = project.root + f'/Files/Self-matching/ER/{algo}_{ds}_nA={nA:d}.csv'

# Check existence
if os.path.exists(fname) and not force:
  sys.exit()
  
pa.line(f'ER - {algo}')

# Creating dataframe
df = pd.DataFrame(columns=['p', 'nRun', 'g', 'q', 'g_std', 'q_std'])

k = 0

ref = time.time()

for p in l_p:

  print(f'{nRun:d} iterations: p={p:.02f} ...', end='', flush=True)
  start = time.time()

  g = []
  q = []

  for i in range(nRun):

    Ga = Gnp(nA, p, directed=directed)
    Gb, gt = Ga.shuffle()

    # --- FAQ

    C = Comparison(Ga, Gb)
    M = C.get_matching(algorithm=algo)
    M.compute_accuracy(gt)

    g.append(M.accuracy)
    q.append(M.structural_quality)

  # --- Store
    
  # Parameters
  df.loc[k, 'p'] = p
  df.loc[k, 'nRun'] = nRun

  # Mean values
  df.loc[k, 'g'] = np.mean(g)
  df.loc[k, 'q'] = np.mean(q)

  print(np.mean(q))
  print(np.std(q))

  # Standard deviations
  df.loc[k, 'g_std'] = np.std(g)
  df.loc[k, 'q_std'] = np.std(q)
  
  k += 1

  print('{:.02f} sec'.format((time.time() - start)))

print('Total time: {:.02f} sec'.format((time.time() - ref)))

# --- Save
    
print('Saving ...', end='')
start = time.time()

df.to_csv(fname)

print('{:.02f} sec'.format((time.time() - start)))

