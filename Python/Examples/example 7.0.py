import os

from test_suite import *
import project 
from Network import *
from  Comparison import *
import paprint as pa

os.system('clear')

# === Parameters ===========================================================

nA = 7
nB = 7

algo = 'GASM'

# ==========================================================================

# T = test_suite()

# print(T)

NetA = Network(nA)
NetA.Adj = np.zeros((nA,nA), dtype=bool)
NetA.Adj[0,1] = True
NetA.Adj[0,2] = True
NetA.Adj[0,3] = True
NetA.Adj[1,4] = True
NetA.Adj[2,5] = True
NetA.Adj[3,6] = True
NetA.prepare()

NetB = Network(nB)
NetB.Adj = np.zeros((nB,nB), dtype=bool)
NetB.Adj[0,1] = True
NetB.Adj[0,2] = True
NetB.Adj[0,3] = True
NetB.Adj[1,4] = True
NetB.Adj[2,5] = True
NetB.Adj[3,6] = True
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