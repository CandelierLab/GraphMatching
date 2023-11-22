import os
import matplotlib.pyplot as plt

import project 
from Network import *
from  Comparison import *
import paprint as pa

os.system('clear')

# === Parameters ===========================================================

nA = 5
nB = 5

algo = 'GASP'

# ==========================================================================

NetA = Network(nA)
NetA.Adj = np.zeros((nA,nA), dtype=bool)
NetA.Adj[0,1] = True
NetA.Adj[0,2] = True
NetA.Adj[1,3] = True
NetA.Adj[2,4] = True
NetA.add_node_attr({'measurable': False, 'values': [0, 1, 0, 0, 0]})
NetA.prepare()

NetB = Network(nB)
NetB.Adj = np.zeros((nB,nB), dtype=bool)
NetB.Adj[0,1] = True
NetB.Adj[0,2] = True
NetB.Adj[1,3] = True
NetB.Adj[2,4] = True
NetB.add_node_attr({'measurable': False, 'values': [0, 0, 0, 0, 0]})
NetB.prepare()

# --- Matching

M = matching(NetA, NetB, algorithm=algo)

# --- Output

pa.line(os.path.basename(__file__))
print(f'{len(M)} Matchings:')

for m in M:
  print(m)