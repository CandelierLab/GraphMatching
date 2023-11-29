import os
import time
import pandas as pd

import project
from Network import *
from  Comparison import *
import paprint as pa

os.system('clear')

# --- Parameters -----------------------------------------------------------

# l_nA = [10, 20, 50, 100, 200]
l_nA = [500, 1000]

p = 0.1

nRun = 1

# --------------------------------------------------------------------------

for nA in l_nA:

  # --- Check

  fname = project.root + f'/Files/Speed/nA={nA}.csv'

  pa.line(f'{nRun} tests with nA={nA}')

  # --- Main loop

  
  t = np.empty(nRun)

  for run in range(nRun):

    Net = Network(nA)
    Net.set_rand_edges('ER', p)
    
    # Net.add_edge_attr('rand', name='test_edge')
    # Net.add_node_attr('rand', name='test_node')

    Set, Icor = Net.shuffle()

    start = time.process_time()
    
    M = matching(Net, Set)

    t[run] = time.process_time() - start


  # --- Save

  T = pd.DataFrame()
  T[0] = t
  T.to_csv(fname)