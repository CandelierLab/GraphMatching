import os

import project
from Network import *
from  Comparison import *

import paprint as pa

os.system('clear')

# === Parameters ===========================================================

nA = 5
# p = 0.9

algo = 'FAQ'

delta = 1

np.random.seed(0)

# --------------------------------------------------------------------------

p = np.log(nA)/nA

print('p', p)

# ==========================================================================

# --- Random graphs

Ga = Network(nA)
Ga.set_rand_edges('ER', p_edges=p)

# # Ga.add_edge_attr('rand', name='test_edge')
# # Ga.add_node_attr('rand', name='test_node')

# Ga = Network(nx=nx.circular_ladder_graph(5))

# Ga = star_branched(9,10)

Gb = Ga.degrade(1)

pa.matrix(Ga.Adj, highlight=Ga.Adj!=Gb.Adj)

C = Comparison(Ga, Gb)
M = C.get_matching(algorithm=algo)
M.compute_accuracy()

print(M)
# pa.matrix(C.X)

print(Ga.nNd, Ga.nEd)
