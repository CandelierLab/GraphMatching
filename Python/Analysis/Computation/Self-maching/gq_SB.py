'''
Star-branched graph: average gamma and q
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

l_k = np.arange(2,11)
l_n = np.arange(1,11)

l_eta = [1e-3]
nRun = 1000

# ==========================================================================

fname = project.root + f'/Files/Self-matching/SB/nRun={nRun:d}.csv'

# Creating dataframe
df = pd.DataFrame(columns=['k', 'n', 'eta', 'nRun', 'g_Zager', 'g_GASM', 'q_Zager', 'q_GASM', 'g_Zager_std', 'g_GASM_std', 'q_Zager_std', 'q_GASM_std'])

i = 0

for k in l_k:
  for n in l_n:

    print(f'k={k:d}, n={n:d}, ', end='')

    NetA = star_branched(k,n, directed=True)

    for eta in l_eta:

      print(f'{nRun:d} iterations: eta={eta:.05f} ...', end='')
      start = time.time()
      
      g_Zager = []
      g_GASM = []
      q_Zager = []
      q_GASM = []

      for j in range(nRun):

        NetB, Idx = NetA.shuffle()

        # --- Zager

        C = Comparison(NetA, NetB)
        M = C.get_matching(algorithm='Zager')
        M.compute_accuracy(Idx)

        g_Zager.append(M.accuracy)
        q_Zager.append(M.structural_quality)

        # --- GASM

        C = Comparison(NetA, NetB)
        M = C.get_matching(algorithm='GASM', eta=eta)
        M.compute_accuracy(Idx)

        g_GASM.append(M.accuracy)
        q_GASM.append(M.structural_quality)

      # --- Store
        
      # Parameters
      df.loc[i, 'k'] = k
      df.loc[i, 'n'] = n
      df.loc[i, 'eta'] = eta

      # Mean values
      df.loc[i, 'g_Zager'] = np.mean(g_Zager)
      df.loc[i, 'q_Zager'] = np.mean(q_Zager)
      df.loc[i, 'g_GASM'] = np.mean(g_GASM)
      df.loc[i, 'q_GASM'] = np.mean(q_GASM)

      # Standard deviations
      df.loc[i, 'g_Zager_std'] = np.std(g_Zager)
      df.loc[i, 'q_Zager_std'] = np.std(q_Zager)
      df.loc[i, 'g_GASM_std'] = np.std(g_GASM)
      df.loc[i, 'q_GASM_std'] = np.std(q_GASM)

      i += 1

      print('{:.02f} sec'.format((time.time() - start)))

# --- Save
    
print(i)

print('Saving ...', end='')
start = time.time()

df.to_csv(fname)

print('{:.02f} sec'.format((time.time() - start)))

