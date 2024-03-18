import os

import project
from Network import *
from  Comparison import *

import paprint as pa

os.system('clear')

# --- Parameters -----------------------------------------------------------

nA = 5
p = 0.5

algo = 'Zager'

# np.random.seed(0)

# --------------------------------------------------------------------------

# --- Random graphs

Ga = Network(nA)
Ga.set_rand_edges('ER', p_edges=p)

# # Ga.add_edge_attr('rand', name='test_edge')
# # Ga.add_node_attr('rand', name='test_node')

# Ga = Network(nx=nx.circular_ladder_graph(5))

Gb, Idx = Ga.shuffle()

C = Comparison(Ga, Gb, algorithm=algo)
M = C.get_matching(normalization=1, deg_norm=True)
M.compute_accuracy(Idx)

print(M)
pa.matrix(Ga.Adj)
pa.matrix(C.X)

