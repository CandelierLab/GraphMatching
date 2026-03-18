import os, sys

import numpy as np
import networkx  as nx
import matplotlib.pyplot as plt

import paprint as pa
from Graph import Graph, Gnm

os.system('clear')

# ═══ Parameters ═══════════════════════════════════════════════════════════

# n = 20
# m = 2*n

# n = 100
# m = 2*n

# ══════════════════════════════════════════════════════════════════════════

rng = np.random.default_rng(0)

# G = Gnm(n, m, directed=False, rng=rng)
G = Graph(nx=nx.relaxed_caveman_graph(2, 4, 0.1, seed=1))
n = G.nV

# pa.matrix(G.Adj)

A = G.Adj + np.eye(n, dtype=int)

# ─── Scores ───────────────────────────────────────────────────────────────

# Compute scores
X = np.ones((n))
# Y = np.ones((m))

# X = [d[1] for d in G.nx.degree]

t = n
traces = np.zeros((n,t))
traces[:,0] = X

for i in range(t-1):

  X = X @ A

  X /= np.min(X)

  traces[:,i+1] = X

# print(traces)

# plt.style.use('dark_background')

# plt.plot(traces.T, '.-')

# plt.yscale('log')

# plt.show()

# sys.exit()

# ─── Contrast

print(f'min X: {np.min(X)} - max X: {np.max(X)}')

# ─── Regions ──────────────────────────────────────────────────────────────

# ─── Watershed

# List vertices
I = np.argsort(-X)

# Clusters
C = np.full(n, fill_value=-1, dtype=int)

for i in I:

  # print('---', i)

  # Neighbors
  J = np.where(G.Adj[i,:])[0]

  # Remove unset neighbors
  J = J[C[J]>=0]

  # Neighboring clusters
  K = C[J]

  # print(i, J, K)

  if J.size:
    ''' Join region '''
    
    # # # # Join region with the most neighbors
    # # # K_, counts = np.unique(K, return_counts=True)
    # # # I = counts==np.max(counts)
    # # # if I.size>1:
    # # #   J_ = [j.item() for j in J if C[j] in K_[I]]
    # # #   print(J_)
    # # #   C[i] =  C[J_[np.argmax(X[J_])]]
    # # # else:
    # # #   C[i] = K_[I]

    C[i] = K[np.argmax(X[J])]

  else:
    ''' New region '''
    C[i] = np.max(C)+1

# ─── Summary

U = np.unique(C)
print(f'{U.size} clusters:')
for c in U:
  print(f'[{c}] {np.count_nonzero(C==c)} vertices')

# ─── Display ──────────────────────────────────────────────────────────────

p0 = {i:(C[i].item()+rng.random()*0.01, rng.random()) for i in range(n)}

# ─── Graph ─────────────────────────────────────

fig, ax = plt.subplots(1,2, figsize=(15,7))
G.display(values=X, ax=ax[0])
G.display(values=C, ax=ax[1], cm=plt.cm.tab20)

# plt.style.use('dark_background')

# plt.plot(traces.T, '.-')

# plt.yscale('log')

plt.show()
