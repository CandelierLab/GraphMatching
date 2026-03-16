import os

import numpy as np
import networkx  as nx

import paprint as pa
from Graph import Graph, Gnm

os.system('clear')

# ═══ Parameters ═══════════════════════════════════════════════════════════

n = 20
m = 2*n

# ══════════════════════════════════════════════════════════════════════════

rng = np.random.default_rng(0)

G = Gnm(n, m, directed=False, rng=rng)

# pa.matrix(G.Adj)

# Compute scores
X = np.ones((n))
Y = np.ones((m))

X = [d[1] for d in G.nx.degree]

# ────────────────────────────────────────────────────────────────────────
# Display

G.display(values=X)
