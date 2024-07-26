'''
Degradation of ER: average gamma and q
'''

import os
import numpy as np
import pandas as pd
import time
from alive_progress import alive_bar

import project
from Graph import *
from Comparison import *

os.system('clear')

# === Parameters ===========================================================

directed = True

nA = 200

nRun = 1000

l_delta = np.linspace(0, 1, 21)

# --------------------------------------------------------------------------

# p = np.log(nA)/nA
p = 2/nA

ds = 'directed' if directed else 'undirected'

# ==========================================================================

fname = project.root + f'/Files/Subgraph/ER/{ds}_zeta=0_nA={nA:d}_nRun={nRun:d}.csv'

# --- Generative function
def get_Nets(nA, p, delta):
  
  Ga = Gnp(nA, p, directed=directed)

  # Degradation: remove edges
  Gb, gt = Ga.degrade('vx_rm', delta)

  return (Ga, Gb, gt)

# Creating dataframe
df = pd.DataFrame(columns=['algo', 'delta', 'g', 'q', 'g_std', 'q_std'])

k = 0

for d in l_delta:

  g_FAQ = []
  q_FAQ = []
  g_GASM = []
  q_GASM = []

  with alive_bar(nRun) as bar:
    bar.title = f'delta={d:0.2f}'

    for i in range(nRun):

      Ga, Gb, gt = get_Nets(nA, p, d)
        
      # --- FAQ

      C = Comparison(Ga, Gb)
      M = C.get_matching(algorithm='FAQ')
      M.compute_accuracy(gt)

      g_FAQ.append(M.accuracy)
      q_FAQ.append(M.structural_quality)

      # --- GASM

      C = Comparison(Ga, Gb)
      M = C.get_matching(algorithm='GASM')
      M.compute_accuracy(gt)

      g_GASM.append(M.accuracy)
      q_GASM.append(M.structural_quality)

      bar()

  # Parameters
  df.loc[k, 'algo'] = 'FAQ'
  df.loc[k, 'delta'] = d

  # Mean values
  df.loc[k, 'g'] = np.mean(g_FAQ)
  df.loc[k, 'q'] = np.mean(q_FAQ)

  # Standard deviations
  df.loc[k, 'g_std'] = np.std(g_FAQ)
  df.loc[k, 'q_std'] = np.std(q_FAQ)

  k += 1

  # Parameters
  df.loc[k, 'algo'] = 'GASM'
  df.loc[k, 'delta'] = d

  # Mean values
  df.loc[k, 'g'] = np.mean(g_GASM)
  df.loc[k, 'q'] = np.mean(q_GASM)

  # Standard deviations
  df.loc[k, 'g_std'] = np.std(g_GASM)
  df.loc[k, 'q_std'] = np.std(q_GASM)

  k += 1

# --- Save
    
print('Saving ...', end='')
start = time.time()

df.to_csv(fname)

print('{:.02f} sec'.format((time.time() - start)))

