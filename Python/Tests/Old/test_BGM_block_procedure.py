import os
import time
import numpy as np
import sparse

import project
from Graph import *
import paprint as pa

os.system('clear')

# --- Parameters -----------------------------------------------------------

# np.random.seed(seed=0)

a = 20
b = a

p = 0.95

# --------------------------------------------------------------------------

# Bipartite matrix
A = np.random.rand(a,b) + np.eye(a,b) > p

pa.matrix(A)

# --- Recursive ones assignation

# Sure matches 
sm = np.full(a, None)

I = np.arange(a)
J = np.arange(b)

no_rows = False
no_cols = False

while True:

  # --- Rows

  # Indices
  I_ = np.where(np.sum(A, axis=1)==1)[0]
  J_ = [np.where(A[i,:])[0][0] for i in I_]
  no_rows = len(I_)==0
  
  if no_rows and no_cols: break

  # Update sure matches
  sm[I[I_]] = J[J_]

  # Reduce matrix
  A = np.delete(A, I_, axis=0)
  if A.size:
    A = np.delete(A, J_, axis=1)

  if not A.size: break

  # Reduce index lists
  I = np.delete(I, I_)
  J = np.delete(J, J_)
    
  # pa.matrix(A)
  # print(sm)
  # print(I, J)

  # --- Columns

  # Indices
  J_ = np.where(np.sum(A, axis=0)==1)[0]
  I_ = [np.where(A[:,j])[0][0] for j in J_]
  no_cols = len(I_)==0

  if no_rows and no_cols: break

  # Update sure matches
  sm[I[I_]] = J[J_]

  # Reduce matrix
  A = np.delete(A, I_, axis=0)
  if A.size:
    A = np.delete(A, J_, axis=1)

  if not A.size: break

  # Reduce index lists
  I = np.delete(I, I_)
  J = np.delete(J, J_)

pa.matrix(A)
print('Sure matches:', sm)
# print(I, J)

# --- Separate in blocks

pa.line('Block separation')

# Initialization
U = np.array([0], dtype=int)
V = np.array([], dtype=int)

# Break condition
no_rows = False
no_cols = False

# Search block members
while True:

  # Column mates
  V_ = np.setdiff1d(np.where(np.sum(A[U,:], axis=0))[0], V)
  no_cols = not V_.size
  if no_rows and no_cols: break
  V = np.hstack((V, V_))

  # Row mates
  U_ = np.setdiff1d(np.where(np.sum(A[:,V], axis=1))[0], U)
  no_rows = not U_.size
  if no_rows and no_cols: break
  U = np.hstack((U, U_))

print(U)