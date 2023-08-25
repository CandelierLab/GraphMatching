import os
from Network import *
from  Comparison import *

os.system('clear')

# --- Parameters -----------------------------------------------------------

n = 100
p = 0.1

# --------------------------------------------------------------------------

Net = Network(n)
Net.set_rand_edges('ER', p)

Sub, Icor = Net.subnet(n)

M = matching(Net, Sub, nIter=10, verbose=True)

# Correct matches
print(np.count_nonzero([Icor[m[1]]==m[0] for m in M])/Sub.nNd)