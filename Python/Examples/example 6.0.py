import os

from test_suite import *
import project 
from Network import *
from  Comparison import *
import paprint as pa

os.system('clear')

# === Parameters ===========================================================

nA = 4
nB = 4

algo = 'GASP'

# ==========================================================================

# T = test_suite()

# print(T)

NetA = Network(nA)
NetA.Adj = np.zeros((nA,nA), dtype=bool)
NetA.Adj[0,1] = True
NetA.Adj[1,0] = True
NetA.Adj[1,2] = True
NetA.Adj[2,1] = True
NetA.Adj[2,3] = True
NetA.Adj[3,2] = True
NetA.Adj[0,3] = True
NetA.Adj[3,0] = True
NetA.prepare()

NetB = Network(nB)
NetB.Adj = np.zeros((nB,nB), dtype=bool)
NetB.Adj[0,1] = True
NetB.Adj[1,0] = True
NetB.Adj[1,2] = True
NetB.Adj[2,1] = True
NetB.Adj[2,3] = True
NetB.Adj[3,2] = True
NetB.Adj[0,3] = True
NetB.Adj[3,0] = True
NetB.prepare()

# --- Matching

X, Y = compute_scores(NetA, NetB, algorithm=algo)

pa.matrix(X)

pa.matrix(Y)

# M = matching(NetA, NetB, algorithm=algo)


# --- Output

# pa.line(os.path.basename(__file__))
# print(M)