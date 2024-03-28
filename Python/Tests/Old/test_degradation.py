import os
from Graph import *
from  Comparison import *

os.system('clear')

# --- Parameters -----------------------------------------------------------

n = 5
p = 10

# --------------------------------------------------------------------------

Net = Network(n)
Net.set_rand_edges('ER', p)

Det, Jcor = Net.degrade('struct', n=1)

Net.print()
print(Jcor, np.setdiff1d(range(n), Jcor))
Det.print()

X, Y = scores(Net, Det)

M = matching(Net, Det, nIter=100, verbose=True)

print(M)

# Correct matches
print(np.count_nonzero([[m[1]]==m[0] for m in M])/n)