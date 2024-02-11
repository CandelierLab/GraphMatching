import os
import numpy as np
import math

import project
from BG import Block
import paprint as pa

os.system('clear')

# === Parameters ===========================================================

a = 4

np.random.seed(3)

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

q = a*3

# Base matrix
A = (np.eye(a) + np.diag(np.ones(a-1), 1))>0
A[-1,0] = True

# Add extra entries
I = np.where(A==False)
J = I[1]*a + I[0]
np.random.shuffle(J)
for idx in J[0:q-2*a]:
  A[idx%a, idx//a] = True

A = A.T

B = Block(A)

pa.line('Block matrix')
pa.matrix(B.A)

# === Count solutions ======================================================

# --- Brute force

# pa.line('Brute force')

# S = B.brute_force()[0]

# N = np.zeros((a,a), dtype=int)
# for s in S:
#   for i,j in enumerate(s):
#     N[j,i] += 1

# pa.line('Number of solutions')
# pa.matrix(N)

# --- Test

Nh = np.sum(A, axis=0)
Nv = np.sum(A, axis=1)

Z = np.array(A, dtype=float)

for iter in range(20):

  sh = np.sum(Z, axis=0)
  sv = np.sum(Z, axis=1)

  # Line mean
  lm = np.sum(Z)/a

  # Difference
  dv = (sv-lm)/Nv
  dh = (sh-lm)/Nh

  Z = np.multiply(Z - (dv[:,None]/2 + dh/2), A)

  # pa.line(f'{iter+1}')
  # print(lm)

pa.matrix(np.round(Z*6/3))