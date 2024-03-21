'''
Erdo-Renyi: average gamma and q
'''

import os
import numpy as np
import pandas as pd
import time

import project
from Network import *
from Comparison import *

os.system('clear')

# === Parameters ===========================================================

# l_n = np.linspace(0,400,21, dtype=int)
l_n = np.logspace(1,np.log10(400),21, dtype=int)
l_n[0] = 1

eta = 1e-10
nRun = 10

# --------------------------------------------------------------------------

l_p = np.log(l_n)/l_n

# ==========================================================================

fname = project.root + f'/Files/Speed/ER_nRun={nRun:d}.csv'

# Creating dataframe
df = pd.DataFrame(columns=['n', 'FAQ', 'Zager', 'GASM', 'FAQ_std',  'Zager_std', 'GASM_std'])

k = 0

for i, nA in enumerate(l_n):

  print(f'nA={nA:d} - {nRun:d} iterations ...', end='')
  start = time.time()
  
  FAQ = []
  Zager = []
  GASM = []
  GASM_scores = []
  GASM_LAP = []

  for r in range(nRun):

    NetA = Network(nA)
    NetA.set_rand_edges('ER', p_edges=l_p[i])
    NetB, Idx = NetA.shuffle()

    # --- FAQ

    C = Comparison(NetA, NetB)
    M = C.get_matching(algorithm='FAQ')
    M.compute_accuracy(Idx)

    FAQ.append(M.time['total'])

    # --- Zager

    C = Comparison(NetA, NetB)
    M = C.get_matching(algorithm='Zager')
    M.compute_accuracy(Idx)

    Zager.append(M.time['total'])

    # --- GASM

    C = Comparison(NetA, NetB)
    M = C.get_matching(algorithm='GASM', eta=eta)
    M.compute_accuracy(Idx)

    GASM.append(M.time['total'])
    GASM_scores.append(M.time['scores'])
    GASM_LAP.append(M.time['LAP'])

  # --- Store
    
  # Parameters
  df.loc[k, 'n'] = nA

  # Mean values
  df.loc[k, 'FAQ'] = np.mean(FAQ)
  df.loc[k, 'Zager'] = np.mean(Zager)
  df.loc[k, 'GASM'] = np.mean(GASM)
  df.loc[k, 'GASM_scores'] = np.mean(GASM_scores)
  df.loc[k, 'GASM_LAP'] = np.mean(GASM_LAP)

  # Standard deviations
  df.loc[k, 'FAQ_std'] = np.std(FAQ)
  df.loc[k, 'Zager_std'] = np.std(Zager)
  df.loc[k, 'GASM_std'] = np.std(GASM)
  df.loc[k, 'GASM_scores_std'] = np.std(GASM_scores)
  df.loc[k, 'GASM_LAP_std'] = np.std(GASM_LAP)

  k += 1

  print('{:.02f} sec'.format((time.time() - start)))

# --- Save
    
print('Saving ...', end='')
start = time.time()

df.to_csv(fname)

print('{:.02f} sec'.format((time.time() - start)))

