import os
import time
from scipy import sparse

import project
from Network import *
import paprint as pa

os.system('clear')

# --- Parameters -----------------------------------------------------------

n = 200
p = 0.1

np.random.seed(seed=0)

# --------------------------------------------------------------------------

Net = Network(n)
Net.set_rand_edges('ER', p)
m = Net.nEd

As = Net.As
Y = np.ones((m,m))

# --- Conversion to sparse

# tref = time.perf_counter_ns()

# As_ = sparse.csr_matrix(As)

# print('{:.03f} ms'.format((time.perf_counter_ns()-tref)*1e-6))

# tref = time.perf_counter_ns()

# Y_ = sparse.csr_matrix(Y)

# print('{:.03f} ms'.format((time.perf_counter_ns()-tref)*1e-6))

# --- Direct multiplication

tref = time.perf_counter_ns()

Z = As @ Y

print('{:.03f} ms'.format((time.perf_counter_ns()-tref)*1e-6))

# --- Sparse multiplication

# tref = time.perf_counter_ns()

# Z = As_ @ Y

# print('{:.03f} ms'.format((time.perf_counter_ns()-tref)*1e-6))