import os
import time
import numpy as np

import project
from Network import *
import paprint as pa

os.system('clear')

# --- Parameters -----------------------------------------------------------

np.random.seed(seed=0)

# a = 3
# q = 7

a = 4
q = 10

# a = 5
# q = 15

b = a

# --------------------------------------------------------------------------

# Bipartite matrix
A = np.random.rand(a,b) + np.eye(a,b)
B = A >= np.partition(A.flatten(), -q)[-q]

pa.matrix(B)

# Create large sparse matrix

