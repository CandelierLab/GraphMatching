'''
Speed test on ER graphs

- Separate per algorithm
- directed + undirected
- Add 2opt

'''

import os
import numpy as np
import pandas as pd
import time

import project
from Graph import *
from Comparison import *

os.system('clear')

# === Parameters ===========================================================

directed = False

# l_n = np.linspace(0,400,21, dtype=int)
l_n = np.logspace(1,np.log10(400), 21, dtype=int)
l_n[0] = 1

eta = 1e-10
nRun = 10

# --------------------------------------------------------------------------

l_p = np.log(l_n)/l_n

# ==========================================================================

fname = project.root + f'/Files/Speed/ER_nRun={nRun:d}.csv'

# Creating dataframe
df = pd.DataFrame(columns=['n', 'FAQ', 'Zager', 'GASM_CPU','GASM_GPU', 'FAQ_std',  'Zager_std', 'GASM_CPU_std', 'GASM_GPU_std'])

k = 0

for i, nA in enumerate(l_n):

  print(f'nA={nA:d} - {nRun:d} iterations ...', end='')
  start = time.time()
  
  FAQ = []
  Zager = []
  GASM_CPU = []
  GASM_GPU = []

  for r in range(nRun):

    Ga = Gnp(nA, l_p[i], directed=directed)
    Gb, gt = Ga.shuffle()

    # --- FAQ

    C = Comparison(Ga, Gb)
    M = C.get_matching(algorithm='FAQ')
    M.compute_accuracy(gt)

    FAQ.append(M.time['total'])

    # --- Zager

    C = Comparison(Ga, Gb)
    M = C.get_matching(algorithm='Zager')
    M.compute_accuracy(gt)

    Zager.append(M.time['total'])

    # --- GASM (CPU)

    C = Comparison(Ga, Gb)
    M = C.get_matching(algorithm='GASM', GPU=False)
    M.compute_accuracy(gt)

    GASM_CPU.append(M.time['total'])

    # --- GASM (GPU)

    C = Comparison(Ga, Gb)
    M = C.get_matching(algorithm='GASM', GPU=True)
    M.compute_accuracy(gt)

    GASM_GPU.append(M.time['total'])


  # --- Store
    
  # Parameters
  df.loc[k, 'n'] = nA

  # Mean values
  df.loc[k, 'FAQ'] = np.mean(FAQ)
  df.loc[k, 'Zager'] = np.mean(Zager)
  df.loc[k, 'GASM_CPU'] = np.mean(GASM_CPU)
  df.loc[k, 'GASM_GPU'] = np.mean(GASM_GPU)

  # Standard deviations
  df.loc[k, 'FAQ_std'] = np.std(FAQ)
  df.loc[k, 'Zager_std'] = np.std(Zager)
  df.loc[k, 'GASM_CPU_std'] = np.std(GASM_CPU)
  df.loc[k, 'GASM_GPU_std'] = np.std(GASM_GPU)

  k += 1

  print('{:.02f} sec'.format((time.time() - start)))

# --- Save
    
print('Saving ...', end='')
start = time.time()

df.to_csv(fname)

print('{:.02f} sec'.format((time.time() - start)))

