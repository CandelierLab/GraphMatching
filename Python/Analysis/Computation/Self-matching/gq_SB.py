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

import paprint as pa

os.system('clear')

# === Parameters ===========================================================

algo = 'FAQ'
# algo = '2opt'
# algo = 'Zager'
# algo = 'GASM'

directed = False

# l_k = np.arange(2,11)
# l_k = np.arange(2,6)
l_k = [3]
l_n = np.arange(1,11)

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
fname = project.root + f'/Files/Self-matching/SB/{algo}_{ds}_k=3.csv'

# Check existence
if os.path.exists(fname) and not force:
  sys.exit()
  
pa.line(f'SB - {algo}')

# Creating dataframe
df = pd.DataFrame(columns=['n', 'k', 'nRun', 'g', 'q', 'g_std', 'q_std'])

i = 0
ref = time.time()

for k in l_k:
  for n in l_n:

    print(f'k={k:d}, n={n:d}, {nRun:d} iterations ...', end='', flush=True)
    start = time.time()

    Ga = star_branched(k, n, directed=directed)
      
    g = []
    q = []

    for j in range(nRun):

      Gb, gt = Ga.shuffle()

      # --- FAQ

      C = Comparison(Ga, Gb)
      # M = C.get_matching(algorithm=algo, eta=1e-3)
      M = C.get_matching(algorithm=algo)
      M.compute_accuracy(gt)
      
      g.append(M.accuracy)
      q.append(M.structural_quality)

    # --- Store
      
    # Parameters
    df.loc[i, 'k'] = k
    df.loc[i, 'n'] = n
    df.loc[i, 'nRun'] = nRun

    # Mean values
    df.loc[i, 'g'] = np.mean(g)
    df.loc[i, 'q'] = np.mean(q)

    # Standard deviations
    df.loc[i, 'g_std'] = np.std(g)
    df.loc[i, 'q_std'] = np.std(q)

    i += 1

    print('{:.02f} sec'.format((time.time() - start)))

print('Total time: {:.02f} sec'.format((time.time() - ref)))

# --- Save
    
print('Saving ...', end='')
start = time.time()

df.to_csv(fname)

print('{:.02f} sec'.format((time.time() - start)))

