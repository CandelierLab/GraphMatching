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

# === First exemple ========================================================

pa.line('First exemple (a)')
print()

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
NetB.add_node_attr({'measurable': False, 'values': [0, 1, 0, 0, 0]})
NetB.prepare()

# --- Zager

print('Zager')

X = compute_scores(NetA, NetB, algorithm='Zager', normalization=1, nIter=4)[0]
M = matching(NetA, NetB, algorithm='Zager', normalization=1, nIter=4)

H = np.zeros((nA, nB))
for m in M.matchings:
  for i,j in enumerate(m.J):
    H[i,j] = 1

pa.matrix(X, highlight=H)

print(M)

# --- GASP

print('GASP')

X = compute_scores(NetA, NetB, algorithm='GASP', normalization=1, nIter=4)[0]

M = matching(NetA, NetB, algorithm='GASP', normalization=1, nIter=4)

H = np.zeros((nA, nB))
for m in M.matchings:
  for i,j in enumerate(m.J):
    H[i,j] = 1

pa.matrix(X, highlight=H)

print(M)

# === Second exemple =======================================================

pa.line('Second exemple (b)')
print()

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

# --- Zager

print('Zager')

X = compute_scores(NetA, NetB, algorithm='Zager', normalization=1, nIter=4)[0]
M = matching(NetA, NetB, algorithm='Zager', normalization=1, nIter=4)

H = np.zeros((nA, nB))
for m in M.matchings:
  for i,j in enumerate(m.J):
    H[i,j] = 1

pa.matrix(X, highlight=H)

print(M)

# --- GASP

print('GASP')

X = compute_scores(NetA, NetB, algorithm='GASP', normalization=1, nIter=4)[0]

M = matching(NetA, NetB, algorithm='GASP', normalization=1, nIter=4)

H = np.zeros((nA, nB))
for m in M.matchings:
  for i,j in enumerate(m.J):
    H[i,j] = 1

pa.matrix(X, highlight=H)

print(M)