import os
import numpy as np
import time
from scipy.optimize import quadratic_assignment

import project
from Network import *
from Comparison import *

os.system('clear')

# === Parameters ===========================================================

nA = 50
Nsub = list(range(1, nA+1))

p = np.log(nA)/nA

# --------------------------------------------------------------------------

# ==========================================================================

# --- Definitions

NetA = Network(nA)
NetA.set_rand_edges('ER', p)

NetB = NetA
# NetB, Idx = NetA.shuffle()

NetA.print()
NetB.print()

# print(Idx)

# --- FAQ

print('FAQ ...', end='')
start = time.time()

res = quadratic_assignment(NetA.Adj, NetB.Adj)

print('{:.02f} sec'.format((time.time() - start)))

print(res.col_ind)


# --- GASP

# print('GASP ...', end='')
# start = time.time()

# M = matching(NetA, NetB)

# # Correct matches
# gamma = np.count_nonzero([Idx[m[1]]==m[0] for m in M])/nA

# print('{:.02f} sec'.format((time.time() - start)))

# print(gamma)

