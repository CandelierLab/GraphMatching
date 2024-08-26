import os
import pandas as pd

import project
from Graph import *
from  Comparison import *
# from alive_progress import alive_bar

os.system('clear')

# === Parameters ===========================================================

directed = False

# l_nA = [10, 20, 50, 100, 200]
l_nA = [20]

l_p = np.linspace(0, 1, 101)

nRun = 100

force = True

# --------------------------------------------------------------------------

ds = 'directed' if directed else 'undirected'

# ==========================================================================

for nA in l_nA:

  # --- Check

  fname = project.root + f'/Files/k_star/{ds}_nA={nA}.csv'

  if os.path.exists(fname) and not force:
    continue

  pa.line(f'{nRun} graphs with nA={nA}')

  # --- Main loop

  df = pd.DataFrame(columns=['p', 'kstar', 'kstar_std'])

  for i, p in enumerate(l_p):

    K = []

    for run in range(nRun):

      Net = Gnp(nA, p, directed)      
      K.append(0 if Net.diameter is None else Net.diameter)
     
    # --- Store

    df.loc[i, 'p'] = p
    df.loc[i, 'kstar'] = np.mean(K)
    df.loc[i, 'kstar_std'] = np.std(K)

  # --- Save
  df.to_csv(fname)