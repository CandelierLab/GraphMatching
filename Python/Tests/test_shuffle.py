import os

import project
from Network import *
from  Comparison import *

import paprint as pa

os.system('clear')

# === Parameters ===========================================================

nA = 200
# p = 0.9

algo = 'GASM'

# np.random.seed(0)

# --------------------------------------------------------------------------

p = np.log(nA)/nA

# ==========================================================================

# --- Random graphs

Ga = Network(nA)
Ga.set_rand_edges('ER', p_edges=p)

# # Ga.add_edge_attr('rand', name='test_edge')
# # Ga.add_node_attr('rand', name='test_node')

# Ga = Network(nx=nx.circular_ladder_graph(5))

# Ga = star_branched(9,10)

Gb, Idx = Ga.shuffle()

C = Comparison(Ga, Gb, verbose=True)
M = C.get_matching(algorithm=algo)
M.compute_accuracy(Idx)

print(M)
# pa.matrix(C.X)

print(Ga.nNd, Ga.nEd)
