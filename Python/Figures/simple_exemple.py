import os
import matplotlib.pyplot as plt

import project
from Network import *
from  Comparison import *

os.system('clear')

# === Parameters ===========================================================

nA = 5
nB = 5

nIter = 4

# ==========================================================================

NetA = Network(nA)
NetA.Adj = np.zeros((nA,nA), dtype=bool)
NetA.Adj[0,1] = True
NetA.Adj[0,2] = True
NetA.Adj[1,3] = True
NetA.Adj[2,4] = True
NetA.add_node_attr({'measurable': False, 'values': [0, 0, 0, 0, 0]})
NetA.prepare()

NetB = Network(nB)
NetB.Adj = np.zeros((nB,nB), dtype=bool)
NetB.Adj[0,1] = True
NetB.Adj[0,2] = True
NetB.Adj[1,3] = True
NetB.Adj[2,4] = True
NetB.add_node_attr({'measurable': False, 'values': [0, 1, 0, 0, 0]})
NetB.prepare()


NetA.print()

X, Y = scores(NetA, NetB, nIter=nIter, normalization=1)

print(X)

M = matching(NetA, NetB, nIter=nIter, normalization=1)

print(M)

# # # R = np.zeros(nIterMax, dtype=int)

# # # for i in range(nIterMax):

# # #   M = matching(NetA, NetB, nIter=i+1)

# # #   for k, m in enumerate(M):
# # #      R[i] += int(m[1]!=M0[k][1])

# # #   # Update reference
# # #   M0 = M

# # # # === Display =================================================================

# # # plt.style.use('dark_background')

# # # fig, ax = plt.subplots()

# # # ax.plot(R, '.-')

# # # ax.set_xlabel('Iterations')
# # # ax.set_ylabel('Number of changes')

# # # plt.show()