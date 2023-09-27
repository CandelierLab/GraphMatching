import numpy as np
from scipy.optimize import quadratic_assignment

# Parameters
n = 5
p = np.log(n)/n

# Definitions
A = np.random.rand(n,n)<p

# Quadratic assignmeent
res = quadratic_assignment(A, A)

print(res.col_ind)

print(p)
