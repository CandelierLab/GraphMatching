import os
import matplotlib.pyplot as plt

import project 
from Graph import *
from  Comparison import *
import paprint as pa

os.system('clear')

# === Parameters ===========================================================

directed = True

nA = 5
nB = 5

nIter = 2

np.random.seed(0)

# === First exemple ========================================================

pa.line('First exemple')
print()

Adj = np.zeros((nA,nA), dtype=bool)
Adj[0,1] = True
Adj[0,2] = True
Adj[1,3] = True
Adj[2,4] = True
Ga = Graph(nA, directed=directed, Adj=Adj)

Adj = np.zeros((nB,nB), dtype=bool)
Adj[0,1] = True
Adj[0,2] = True
Adj[1,3] = True
Adj[2,4] = True
Gb = Graph(nB, directed=directed, Adj=Adj)

gt = GroundTruth(Ga, Gb)

# --- Zager

print('Zager')

C = Comparison(Ga, Gb)
M = C.get_matching(algorithm='Zager', normalization=1, nIter=nIter)
# M.compute_accuracy(gt)

pa.matrix(C.X)

# --- GASM

print('GASM')

C = Comparison(Ga, Gb)
M = C.get_matching(algorithm='GASM', normalization=1, nIter=nIter, eta=1e-2)
# M.compute_accuracy(gt)

print(C.X)

# === Second exemple ========================================================

pa.line('Second exemple')
print()

Adj = np.zeros((nA,nA), dtype=bool)
Adj[0,1] = True
Adj[0,2] = True
Adj[1,3] = True
Adj[2,4] = True
Ga = Graph(nA, directed=directed, Adj=Adj)
Ga.add_vrtx_attr({'precision': 0, 'values': [0, 1, 0, 0, 0]})

Adj = np.zeros((nB,nB), dtype=bool)
Adj[0,1] = True
Adj[0,2] = True
Adj[1,3] = True
Adj[2,4] = True
Gb = Graph(nB, directed=directed, Adj=Adj)
Gb.add_vrtx_attr({'precision': 0, 'values': [0, 1, 0, 0, 0]})

gt = GroundTruth(Ga, Gb)

# --- Zager

print('Zager')

C = Comparison(Ga, Gb)
M = C.get_matching(algorithm='Zager', normalization=1, nIter=nIter)
# M.compute_accuracy(gt)

pa.matrix(C.X)

# --- GASM

print('GASM')

C = Comparison(Ga, Gb)
M = C.get_matching(algorithm='GASM', normalization=1, nIter=nIter)
# M.compute_accuracy(gt)

pa.matrix(C.X)


# === Third exemple ========================================================

pa.line('Third exemple')
print()

Adj = np.zeros((nA,nA), dtype=bool)
Adj[0,1] = True
Adj[0,2] = True
Adj[1,3] = True
Adj[2,4] = True
Ga = Graph(nA, directed=directed, Adj=Adj)
Ga.add_vrtx_attr({'precision': 0, 'values': [0, 0, 0, 0, 0]})

Adj = np.zeros((nB,nB), dtype=bool)
Adj[0,1] = True
Adj[0,2] = True
Adj[1,3] = True
Adj[2,4] = True
Gb = Graph(nB, directed=directed, Adj=Adj)
Gb.add_vrtx_attr({'precision': 0, 'values': [0, 1, 0, 0, 0]})

gt = GroundTruth(Ga, Gb)

# --- Zager

print('Zager')

C = Comparison(Ga, Gb)
M = C.get_matching(algorithm='Zager', normalization=1, nIter=nIter)
# M.compute_accuracy(gt)

pa.matrix(C.X)

# --- GASM

print('GASM')

C = Comparison(Ga, Gb)
M = C.get_matching(algorithm='GASM', normalization=1, nIter=nIter, eta=1e-2)
# M.compute_accuracy(gt)

print(C.X)

# print(M)
