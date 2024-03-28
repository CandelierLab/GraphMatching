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

nA = 100
nRun = 200

# l_delta = np.geomspace(0.01, 0.5, 10)
l_delta = np.linspace(0, 1, 21)

l_meas = [0, 1, 2, 3, 5]

# --------------------------------------------------------------------------

p = np.log(nA)/nA
# p = 0.25
# print(p, p*nA**2)

# ==========================================================================

fname = project.root + f'/Files/Degradation/ER/nA={nA:d}_nRun={nRun:d}.csv'

# --- Generative function
def get_Nets(nA, p, m, d):
  
  NetA = Network(nA)
  NetA.set_rand_edges('ER', p_edges=p)

  for i in range(m):
    NetA.add_node_attr('rand')

  NetB = NetA.degrade(d)

  return (NetA, NetB)


# Creating dataframe
df = pd.DataFrame(columns=['algo', 'delta', 'algo', 'nMeasAttr', 'g', 'q', 'g_std', 'q_std'])

k = 0

for d in l_delta:

  print(f'delta={d:0.2f} - {nRun:d} iterations ...', end='')
  start = time.time()

  g_GASM = [[] for m in l_meas]
  q_GASM = [[] for m in l_meas]

  # --- FAQ

  g = []
  q = []

  for i in range(nRun):

    NetA, NetB = get_Nets(nA, p, 0, d)
      
    C = Comparison(NetA, NetB)
    M = C.get_matching(algorithm='FAQ')
    M.compute_accuracy()

    g.append(M.accuracy)
    q.append(M.structural_quality)

  # Parameters
  df.loc[k, 'algo'] = 'FAQ'
  df.loc[k, 'delta'] = d
  df.loc[k, 'nMeasAttr'] = 0

  # Mean values
  df.loc[k, 'g'] = np.mean(g)
  df.loc[k, 'q'] = np.mean(q)

  # Standard deviations
  df.loc[k, 'g_std'] = np.std(g)
  df.loc[k, 'q_std'] = np.std(q)

  k += 1

  # --- GASM

  for m in l_meas:

    g = []
    q = []

    for i in range(nRun):

      NetA, NetB = get_Nets(nA, p, m, d)
        
      C = Comparison(NetA, NetB)
      M = C.get_matching(algorithm='GASM')
      M.compute_accuracy()

      g.append(M.accuracy)
      q.append(M.structural_quality)

    # Parameters
    df.loc[k, 'algo'] = 'GASM'
    df.loc[k, 'delta'] = d
    df.loc[k, 'nMeasAttr'] = m

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

