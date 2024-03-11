import os
import matplotlib.pyplot as plt

import project 
from Network import *
from  Comparison import *
import paprint as pa

os.system('clear')

# === Parameters ===========================================================

nA = 4
nB = 4

algo = 'GASM'

# ==========================================================================

NetA = Network(nA, directed=False)
NetA.Adj = np.zeros((nA,nA), dtype=bool)
NetA.Adj[0,1] = True
NetA.Adj[1,2] = True
NetA.Adj[2,3] = True
NetA.add_node_attr({'measurable': False, 'values': [0, 1, 0, 0]})
NetA.prepare()

NetB = Network(nB, directed=False)
NetB.Adj = np.zeros((nB,nB), dtype=bool)
NetB.Adj[0,1] = True
NetB.Adj[1,2] = True
NetB.Adj[2,3] = True
NetB.add_node_attr({'measurable': False, 'values': [0, 0, 0, 0]})
NetB.prepare()

# --- Matching

C = Comparison(NetA, NetB, algorithm=algo)

# C.compute_scores()

M = C.get_matching()

# --- Output

pa.line(os.path.basename(__file__))
print()

# pa.matrix(C.X, title='Node scores')
# pa.matrix(C.Y)

print(M)