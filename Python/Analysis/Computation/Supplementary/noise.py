import os, sys
import numpy as np
import pandas as pd
import time

import project
from Graph import *
from Comparison import *

os.system('clear')

# === Parameters ===========================================================

l_directed = [True, False]
l_p = np.linspace(0, 1, 51)

nA = 20
nRun = 1000

l_eta = np.insert(np.logspace(-15, 0, 8), 0, 0)

force = True

dname = project.root + '/Files/Noise/'

# ==========================================================================

for directed in l_directed:

  sdir = 'directed' if directed else 'undirected'

  for eta in l_eta:

    fname = dname + f'ER_{sdir}_nA={nA}_eta={eta}_nRun={nRun}.csv'

    # Skip if existing
    if os.path.exists(fname) and not force: continue

    # Creating dataframe
    df = pd.DataFrame(columns=['g', 'q'])

    print(f'{sdir:s} - eta={eta} ...', end='')
    start = time.time()

    i = 0

    for p in l_p:

      for k in range(nRun):

        Ga = Gnp(nA, p, directed=directed)
        Gb, gt = Ga.shuffle()

        C = Comparison(Ga, Gb)
        M = C.get_matching(algorithm='GASM', GPU=False, eta=eta)
        M.compute_accuracy(gt)

        df.loc[i, 'p'] = p
        df.loc[i, 'g'] = M.accuracy
        df.loc[i, 'q'] = M.structural_quality

        i += 1

    print('{:.02f} sec'.format((time.time() - start)))

    # --- Save

    df.to_csv(fname)