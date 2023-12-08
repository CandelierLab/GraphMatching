import os
import time
import numpy as np

import project
from Network import *
import paprint as pa

os.system('clear')

# --- Parameters -----------------------------------------------------------

np.random.seed(seed=0)

a = 5
b = a

p = 0.9

# --------------------------------------------------------------------------

# Bipartite matrix
A = np.random.rand(a,b) + np.eye(a,b) > p

pa.matrix(A)

# --- Recursive ones assignation

# Sure matches
sm = np.full(a, None)

I = np.arange(a)
J = np.arange(b)

for iter in range(1):

  # --- Rows

  # Indices
  I_ = np.where(np.sum(A, axis=1)==1)[0]
  J_ = [np.where(A[i,:])[0][0] for i in I_]

  # Update sure matches
  sm[I[I_]] = J[J_]

  # Reduce matrix
  A = np.delete(A, I_, axis=0)
  A = np.delete(A, J_, axis=1)

  # Reduce index lists
  I = np.delete(I, I_)
  J = np.delete(J, J_)
    

  pa.matrix(A)
  print(sm)
  print(I, J)

  # --- Columns

  # Indices
  J_ = np.where(np.sum(A, axis=0)==1)[0]
  I_ = [np.where(A[:,j])[0][0] for j in J_]

  # Update sure matches
  sm[I[I_]] = J[J_]

  # Reduce matrix
  A = np.delete(A, I_, axis=0)
  A = np.delete(A, J_, axis=1)

  # Reduce index lists
  I = np.delete(I, I_)
  J = np.delete(J, J_)

  pa.matrix(A)
  print(sm)
  print(I, J)

# --- Separate in blocks


