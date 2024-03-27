import os

import project
from Network import *
from  Comparison import *

import paprint as pa

os.system('clear')

# === Parameters ===========================================================

nA = 5
p = np.log(nA)/nA

algo = 'FAQ'

type = 'ed_rm'
delta = 0.5
preserval = True

# --------------------------------------------------------------------------

np.random.seed(0)

# ==========================================================================

# print('p', p)

# --- Random graphs

Ga = Network(nA)
Ga.set_rand_edges('ER', p_edges=p)

print(Ga.nx.is_directed())

Gb = Ga.degrade(type, delta, preserval=preserval)

pa.matrix(Ga.Adj, highlight=Ga.Adj!=Gb.Adj)

C = Comparison(Ga, Gb)
M = C.get_matching(algorithm=algo)
M.compute_accuracy()

print(M)