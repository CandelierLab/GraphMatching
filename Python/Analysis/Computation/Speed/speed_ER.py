'''
Speed test on ER graphs

- Separate per algorithm
- directed + undirected
- Add 2opt

'''

import os, sys
import numpy as np
import pandas as pd
import time

import project
from Graph import *
from Comparison import *

os.system('clear')

# === Parameters ===========================================================

l_directed = [False, True]
# l_algo = ['FAQ', '2opt', 'Zager', 'GASM_CPU', 'GASM_GPU']
# l_n = np.unique(np.logspace(0,np.log10(10000), 101, dtype=int))

l_algo = ['GASM_GPU']
l_n = [1000]

nRun = 10

# Maximal time (s)
mt = 60

force = True

# --------------------------------------------------------------------------

l_p = np.log(l_n)/l_n

# Maximal time per graph
mtpg = mt/nRun

# ==========================================================================

for directed in l_directed:

  sdir = 'directed' if directed else 'undirected'

  for algo in l_algo:

    fname = project.root + f'/Files/Speed/ER_{algo}_{sdir:s}.csv'

    # Skip if already existing
    if os.path.exists(fname) and not force: continue

    # Creating dataframe
    df = pd.DataFrame(columns=['n', 'p', 't'])

    k = 0
    tcheck = False

    for i, nA in enumerate(l_n):

      # Time check
      if tcheck: break

      print(f'{algo} {sdir:s} nA={nA:d} - {nRun:d} iterations ...', end='')
      start = time.time()
      
      for r in range(nRun):

        '''
        !! Set seed !! @ r+nA*nRun
        '''

        Ga = Gnp(nA, l_p[i], directed=directed)
        Gb, gt = Ga.shuffle()

        # --- FAQ

        C = Comparison(Ga, Gb)

        match algo:
          case 'GASM_CPU':
            M = C.get_matching(algorithm='GASM', GPU=False)
          case 'GASM_GPU':
            M = C.get_matching(algorithm='GASM', GPU=True)
          case _:
            M = C.get_matching(algorithm=algo)

        M.compute_accuracy(gt)

        # --- Store
          
        # Parameters
        df.loc[k, 'n'] = nA
        df.loc[k, 'p'] = l_p[i]
        df.loc[k, 't'] = M.time['total']

        k += 1

        # Time check
        if M.time['total']/1000>mtpg:
          tcheck = True

      print(' {:.02f} sec'.format((time.time() - start)))

    print(df)

    # --- Save
        
    # print('Saving ...', end='')
    # start = time.time()

    # df.to_csv(fname)

    # print('{:.02f} sec'.format((time.time() - start)))

