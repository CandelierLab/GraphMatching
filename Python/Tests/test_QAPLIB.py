import os
from scipy.optimize import quadratic_assignment
import matplotlib.pyplot as plt

import project
from Graph import *
from  Comparison import *
from QAPLIB import QAPLIB

import paprint as pa

os.system('clear')

# === Parameters ===========================================================

id = 'esc128'

algo = 'GASM'

np.random.seed(0)

# ==========================================================================

Q = QAPLIB()
I = Q.get(id)

print(I.s)

print('Sol: ', np.trace(I.A.T @ I.B[I.s, :][:, I.s]))

res = quadratic_assignment(I.A, I.B)
P = res['col_ind']
print('FAQ: ', np.trace(I.A.T @ I.B[P, :][:, P]))

# print(P)

Ga, Gb, gt = Q.get_graphs(id)

C = Comparison(Ga, Gb)
M = C.get_matching(algorithm=algo)

sGASM = np.trace(I.A.T @ I.B[M.idxB, :][:, M.idxB])

print('GASM:', sGASM)

# plt.style.use('dark_background')
# fig, ax = plt.subplots(1, 1, figsize=(10,10))

# ax.plot(x, y, '.-')

# plt.show()
