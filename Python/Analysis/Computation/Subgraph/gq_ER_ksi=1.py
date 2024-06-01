'''
Degradation of ER: average gamma and q
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

nA = 200
p = np.log(nA)/nA

nRun = 1000

l_delta = np.linspace(0, 1, 21)
l_precision = np.round(np.geomspace(0.005, 1, 5)*1000)/1000
l_precision = np.r_[0, l_precision]

# --------------------------------------------------------------------------

ds = 'directed' if directed else 'undirected'

# ==========================================================================

fname = project.root + f'/Files/Degradation/ER/{ds}_ksi=1_nA={nA:d}_nRun={nRun:d}.csv'

# --- Generative function

def get_Nets(nA, p, delta):
  
  Ga = Gnp(nA, p, directed=directed)

  # Edge atrribute
  Ga.add_edge_attr('gauss')

  # Degradation: remove edges
  Gb, gt = Ga.degrade('ed_rm', delta)

  return (Ga, Gb, gt)

# Creating dataframe
df = pd.DataFrame(columns=['algo', 'delta', 'g', 'q', 'g_std', 'q_std'])

k = 0

for d in l_delta:

  print(f'delta={d:0.2f} - {nRun:d} iterations ...', end='', flush=True)
  start = time.time()

  # --- Initialization

  g_FAQ = np.empty((nRun))
  q_FAQ = np.empty((nRun))
  g_GASM = np.empty((nRun))
  q_GASM = np.empty((nRun))
  g_GASM_p = np.empty((len(l_precision), nRun))
  q_GASM_p = np.empty((len(l_precision), nRun))

  for i in range(nRun):

    Ga, Gb, gt = get_Nets(nA, p, d)
      
    # --- FAQ

    C = Comparison(Ga, Gb)
    M = C.get_matching(algorithm='FAQ')
    M.compute_accuracy(gt)

    g_FAQ[i] = M.accuracy
    q_FAQ[i] = M.structural_quality

    # --- GASM

    C = Comparison(Ga, Gb)
    M = C.get_matching(algorithm='GASM')
    M.compute_accuracy(gt)

    g_GASM[i] = M.accuracy
    q_GASM[i] = M.structural_quality

    # --- GASM with fixed precision

    for j, precision in enumerate(l_precision):

      Ga.edge_attr[0]['precision'] = precision
      Gb.edge_attr[0]['precision'] = precision

      C = Comparison(Ga, Gb)
      M = C.get_matching(algorithm='GASM')
      M.compute_accuracy(gt)

      g_GASM_p[j,i] = M.accuracy
      q_GASM_p[j,i] = M.structural_quality

  # --- FAQ

  # Parameters
  df.loc[k, 'algo'] = 'FAQ'
  df.loc[k, 'delta'] = d
  df.loc[k, 'precision'] = None

  # Mean values
  df.loc[k, 'g'] = np.mean(g_FAQ)
  df.loc[k, 'q'] = np.mean(q_FAQ)

  # Standard deviations
  df.loc[k, 'g_std'] = np.std(g_FAQ)
  df.loc[k, 'q_std'] = np.std(q_FAQ)

  k += 1

  # --- GASM

  # Parameters
  df.loc[k, 'algo'] = 'GASM'
  df.loc[k, 'delta'] = d
  df.loc[k, 'precision'] = None

  # Mean values
  df.loc[k, 'g'] = np.mean(g_GASM)
  df.loc[k, 'q'] = np.mean(q_GASM)

  # Standard deviations
  df.loc[k, 'g_std'] = np.std(g_GASM)
  df.loc[k, 'q_std'] = np.std(q_GASM)

  k += 1

  # --- GASM with precision

  for j, precision in enumerate(l_precision):

    # Parameters
    df.loc[k, 'algo'] = 'GASM'
    df.loc[k, 'delta'] = d
    df.loc[k, 'precision'] = precision

    # Mean values
    df.loc[k, 'g'] = np.mean(g_GASM_p[j,:])
    df.loc[k, 'q'] = np.mean(q_GASM_p[j,:])

    # Standard deviations
    df.loc[k, 'g_std'] = np.std(g_GASM_p[j,:])
    df.loc[k, 'q_std'] = np.std(q_GASM_p[j,:])

    k += 1

  print('{:.02f} sec'.format((time.time() - start)))

# --- Save
    
print('Saving ...', end='')
start = time.time()

df.to_csv(fname)

print('{:.02f} sec'.format((time.time() - start)))

