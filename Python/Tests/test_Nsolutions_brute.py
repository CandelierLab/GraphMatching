import os
import numpy as np

import project
import paprint as pa

os.system('clear')

# === Parameters ===========================================================

a = 5

q = 22

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

pa.matrix(A)

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

print(len(S))

