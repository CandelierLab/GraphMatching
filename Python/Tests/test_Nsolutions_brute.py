import os
import numpy as np

import project
import paprint as pa

os.system('clear')

# === Parameters ===========================================================

# a = 5
# q = 15

# a = 3
# q = 7

a = 4
q = 9

np.random.seed(0)

# ==========================================================================

# Checks
assert q>=2*a, 'p is too low'
assert q<=a*a, 'p is too high'

# --- Block matrix
'''
Block matrices have at least 2 elements on each row and each column.
'''

# Base matrix
A = (np.eye(a) + np.diag(np.ones(a-1), 1))>0
A[-1,0] = True

# Add extra entries
I = np.where(A==False)
J = I[1]*a + I[0]
np.random.shuffle(J)
for idx in J[0:q-2*a]:
  A[idx%a, idx//a] = True

# --- Reorganize matrix

pa.matrix(A)

# A = A[:,np.argsort(np.sum(A, axis=0))]
# A = A[np.argsort(np.sum(A, axis=1)),:]
# pa.matrix(A)

# --- Count number of solutions (brute force)

def get_subs(M, base=[], idx=None):

  if idx is None:
    idx = range(M.shape[0])

  if M.size==1:
    return [idx] if M[0] else []

  else:
    R = []

    if np.count_nonzero(M[:,0]):

      for j in np.where(M[:,0])[0]:

        M_ = M[np.setdiff1d(np.arange(M.shape[1]),j), 1:]
        idx_ = [idx[i] for i in np.setdiff1d(list(range(len(idx))), j)]

        for s in get_subs(M_, base + [idx[j]], idx_):
          R.append([idx[j]] + s)

    return R

# List all solutions
S = get_subs(A)

print('Number of solutions:', len(S))
pa.line()

# === Count solutions ======================================================

from scipy.sparse import coo_array

I0, J0 = np.where(A)
U = np.empty(0, dtype=int)
V = np.empty(0, dtype=int)

for j in range(a):
  for i in range(a):

    if not A[i,j]: continue

    # Targets
    if j==a-1:
      K = np.logical_and(i!=I0, J0==0)
    else:
      K = np.logical_and(i!=I0, j+1==J0)

    U = np.concatenate((U, np.full(np.count_nonzero(K), fill_value=i+j*a)))
    V = np.concatenate((V, I0[K] + a*J0[K]))

# --- Big sparse matrix
    
B = coo_array((np.ones(U.size, dtype=int), (U,V)), shape=(a**2, a**2)).tocsr()

# # Symmetrify
# rows, cols = B.nonzero()
# B[cols, rows] = B[rows, cols]

# # Convert to csr
# B = B.tocsr()

print(B)

# Find solutions

I = np.arange(a**2, dtype=int)

C = B.copy()

for i in range(a-1):

  C = C @ B
  
  pa.line(f'{i}')
  print(C)
  
print(C.diagonal())

print('Number of solutions:', C.diagonal().sum())
