import os
import time
import numpy as np

os.system('clear')

# --- Parameters -----------------------------------------------------------

n = 1000
p = 0.5

# --------------------------------------------------------------------------

# Define matrix
A = np.random.rand(n,n)
M = (A < p).astype(float)

print('Number of edges:', np.count_nonzero(M))

# Multiplication
tref = time.process_time_ns()

Z = M@M

print('{:.03f} ms'.format((time.process_time_ns()-tref)/1e7))


