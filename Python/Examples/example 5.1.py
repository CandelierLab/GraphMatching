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

algo = 'GASM'

# ==========================================================================

NetA = Network(nA, directed=False)
NetA.Adj = np.zeros((nA,nA), dtype=bool)
NetA.Adj[0,2] = True
NetA.Adj[0,3] = True
NetA.Adj[0,4] = True
NetA.Adj[1,2] = True
NetA.Adj[1,3] = True
NetA.Adj[1,4] = True
NetA.Adj[2,4] = True
NetA.Adj[3,4] = True
NetA.add_node_attr({'measurable': False, 'values': [1, 0, 0, 0, 0]})
NetA.prepare()

NetB = Network(nB, directed=False)
NetB.Adj = np.zeros((nB,nB), dtype=bool)
NetB.Adj[0,2] = True
NetB.Adj[0,3] = True
NetB.Adj[0,4] = True
NetB.Adj[1,2] = True
NetB.Adj[1,3] = True
NetB.Adj[1,4] = True
NetB.Adj[2,4] = True
NetB.Adj[3,4] = True
NetB.add_node_attr({'measurable': False, 'values': [1, 0, 0, 0, 0]})
NetB.prepare()

# --- Matching

C = Comparison(NetA, NetB, algorithm=algo)

# C.compute_scores()

M = C.get_matching()

# --- Output

pa.line(os.path.basename(__file__))
print()

pa.matrix(C.X, title='Node scores')
# pa.matrix(C.Y)

print(M)