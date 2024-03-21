import os
from scipy.optimize import quadratic_assignment
import project
from Network import *
from  Comparison import *

import paprint as pa

os.system('clear')

# --- Parameters -----------------------------------------------------------

nA = 100
p = 0.5

# np.random.seed(0)

# --------------------------------------------------------------------------

# --- Random graphs

Ga = Network(nA)
Ga.set_rand_edges('ER', p_edges=p)

Gb, Idx = Ga.shuffle()

# pa.matrix(Ga.Adj)
# pa.matrix(Gb.Adj)

# res = quadratic_assignment(Ga.Adj, Gb.Adj, options={'maximize': True})

# print(res)

# print(Idx)
# print(res.col_ind)

# print(res.col_ind[Idx])

C = Comparison(Ga, Gb)
M = C.get_matching(algorithm='GASM')
M.compute_accuracy(Idx)

print(M)

M2 = C.get_matching(algorithm='FAQ')
M2.compute_accuracy(Idx)

print(M2)