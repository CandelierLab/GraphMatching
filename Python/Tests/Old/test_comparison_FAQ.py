import os
import numpy as np
import time
from scipy.optimize import quadratic_assignment

import project
from Graph import *
from Comparison import *

os.system('clear')

# === Parameters ===========================================================

nA = 100
Nsub = list(range(1, nA+1))

p = np.log(nA)/nA

# --------------------------------------------------------------------------

# ==========================================================================

# --- Definitions

NetA = Network(nA)
NetA.set_rand_edges('ER', p)

# NetB = NetA
NetB, Idx = NetA.subnet(50)

# NetA.print()
# NetB.print()

# print(Idx)

# --- FAQ

print('FAQ ...', end='')
start = time.time()

res = quadratic_assignment(NetB.Adj, NetA.Adj, options={'maximize': True})

print('{:.02f} sec'.format((time.time() - start)))

# Accuracy
gamma_FAQ = np.count_nonzero([res.col_ind==Idx])/nA

# --- GASP

print('GASP ...', end='')
start = time.time()

M = matching(NetA, NetB)

print('{:.02f} sec'.format((time.time() - start)))

# Accuracy
gamma_GASP = np.count_nonzero([Idx[m[1]]==m[0] for m in M])/nA


print(f'Accruacy FAQ: {gamma_FAQ}')
print(f'Accruacy GASP: {gamma_GASP}')

