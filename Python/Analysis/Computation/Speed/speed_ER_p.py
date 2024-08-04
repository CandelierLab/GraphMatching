'''
Speed test on ER graphs: p
'''

import os, sys
import numpy as np
import pandas as pd
import time
from alive_progress import alive_bar

import project
from Graph import *
from Comparison import *

os.system('clear')

# === Parameters ===========================================================

l_directed = [False, True]
l_algo = ['FAQ', '2opt', 'Zager', 'GASM_CPU', 'GASM_GPU', 'GASM_GPU']
# l_algo = ['GASM_GPU']

l_n = [20, 50, 100]
l_p = np.linspace(0, 1, 21)

nRun = 10

force = True

# ==========================================================================

for directed in l_directed:

  sdir = 'directed' if directed else 'undirected'

  for algo in l_algo:

    fname = project.root + f'/Files/Speed/p/ER_{algo}_{sdir:s}.csv'

    # Skip if already existing
    if os.path.exists(fname) and not force: continue

    # Creating dataframe
    df = pd.DataFrame(columns=['n', 'p', 't'])

    k = 0

    for i, nA in enumerate(l_n):

      print(f'{algo} {sdir:s} nA={nA:d}')

      with alive_bar(l_p.size) as bar:

        bar.title(f'nA={nA}')

        for p in l_p:        
        
          for r in range(nRun+1):

            Ga = Gnp(nA, p, directed=directed)
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

            if r==0: continue

            M.compute_accuracy(gt)

            # --- Store
              
            # Parameters
            df.loc[k, 'n'] = nA
            df.loc[k, 'p'] = p
            df.loc[k, 't'] = M.time['total']

            k += 1

          bar()

    # --- Save
        
    print('Saving ...', end='')
    start = time.time()

    df.to_csv(fname)

    print('{:.02f} sec'.format((time.time() - start)))

