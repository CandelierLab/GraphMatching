import os
import numpy as np
import math

import project
from BG import Block
import paprint as pa

os.system('clear')

# === Parameters ===========================================================

a = 4

np.random.seed(0)

# ==========================================================================

# --- Block Matrix
'''
Block matrices have at least 2 elements on each row and each column.
'''

# --- Full block

# A = np.ones((a,a), dtype=bool)

# --- Full block with one hole

# A = np.ones((a,a), dtype=int) - np.eye(a, dtype=int)

# --- Randomly filled

q = a*2

# Base matrix
A = (np.eye(a) + np.diag(np.ones(a-1), 1))>0
A[-1,0] = True

# Add extra entries
I = np.where(A==False)
J = I[1]*a + I[0]
np.random.shuffle(J)
for idx in J[0:q-2*a]:
  A[idx%a, idx//a] = True

B = Block(A)

pa.line('Block matrix')

pa.matrix(B.A)

# === Count solutions ======================================================

# --- Brute force

pa.line('Brute force')

nS_BF = B.brute_force()[1]

print('\n', 'Number of solutions:', nS_BF)
print()

# --- Uno

# --- RC algorithm
# '''
# Relaxaton-Compression algorithm
# '''

# pa.line('RC algorithm')

# nS_RC, nOp = B.RC()

# pa.line()
# print('Number of solutions:', nS_RC)
# print('Number of operations:', nOp)
# print('Ryser (theoretical): ', a*2**(a-1))
# print()

# --- Elimination
# '''
# Elimination algorithm
# '''

pa.line('Elimination algorithm')

print()

B.elimination()

# pa.line()
# print('Number of solutions:', nS_RC)
# print('Number of operations:', nOp)
# print('Ryser (theoretical): ', a*2**(a-1))
# print()

# X = [[0,1,-1], [1,2,3], [3,2,1], [0,-1,1]]
# C = [0,0,18, 0]

# X = [[1,2,3], [3,2,1]]
# C = [0,18]


# R = np.linalg.lstsq(X,C, rcond=None)
# print(R[0])
# print(R[1])