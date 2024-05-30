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

directed = True

nA = 200
p = np.log(nA)/nA

# Precision
rho_a = 0.01

# nA = 50
# p = 0.25

nRun = 200

# l_delta = np.geomspace(0.01, 0.5, 10)
l_delta = np.linspace(0, 1, 21)

l_vrtx_attr_meas = [0, 1]
l_edge_attr_meas = [0, 1]

# --------------------------------------------------------------------------

# print(p, p*nA**2)

ds = 'directed' if directed else 'undirected'

# ==========================================================================

fname = project.root + f'/Files/Degradation/ER/{ds}_nA={nA:d}_p={p:.05f}_rhoa={rho_a:.02f}_nRun={nRun:d}.csv'

# --- Generative function
def get_Nets(nA, p, nvam, neam, delta):
  
  Ga = Gnp(nA, p, directed=directed)

  for i in range(nvam):
    Ga.add_vrtx_attr('gauss', precision=0.01)

  for i in range(neam):
    Ga.add_edge_attr('gauss', precision=0.01)

  # Degradation: remove edges
  Gb, gt = Ga.degrade('ed_rm', delta)

  return (Ga, Gb, gt)

# Creating dataframe
df = pd.DataFrame(columns=['algo', 'delta', 'algo', 'nMeasAttr', 'g', 'q', 'g_std', 'q_std'])

k = 0

for d in l_delta:

  print(f'delta={d:0.2f} - {nRun:d} iterations ...', end='', flush=True)
  start = time.time()

  for nvam in l_vrtx_attr_meas:

    for neam in l_edge_attr_meas:

      # Just one attribute at a time
      if neam and nvam: continue

      # --- FAQ

      if nvam==0 and neam<=1:

        g = []
        q = []

        for i in range(nRun):

          Ga, Gb, gt = get_Nets(nA, p, nvam, neam, d)
            
          C = Comparison(Ga, Gb)
          M = C.get_matching(algorithm='FAQ')
          M.compute_accuracy(gt)

          g.append(M.accuracy)
          q.append(M.structural_quality)

        # Parameters
        df.loc[k, 'algo'] = 'FAQ'
        df.loc[k, 'delta'] = d
        df.loc[k, 'nvam'] = nvam
        df.loc[k, 'neam'] = neam

        # Mean values
        df.loc[k, 'g'] = np.mean(g)
        df.loc[k, 'q'] = np.mean(q)

        # Standard deviations
        df.loc[k, 'g_std'] = np.std(g)
        df.loc[k, 'q_std'] = np.std(q)

        k += 1

      # --- GASM

      g = []
      q = []

      for i in range(nRun):

        Ga, Gb, gt = get_Nets(nA, p, nvam, neam, d)
          
        C = Comparison(Ga, Gb)
        M = C.get_matching(algorithm='GASM')
        M.compute_accuracy(gt)

        g.append(M.accuracy)
        q.append(M.structural_quality)

      # Parameters
      df.loc[k, 'algo'] = 'GASM'
      df.loc[k, 'delta'] = d
      df.loc[k, 'nvam'] = nvam
      df.loc[k, 'neam'] = neam

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

