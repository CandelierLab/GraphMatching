import os
from Network import *
from  Comparison import *

os.system('clear')

n = 5

Net = Network(n)
Net.set_rand_edges('ER', 0.5)

# --- Identity

Met = copy.deepcopy(Net)
Icor = np.arange(n)

# --- Scrambling

Met, Icor = Net.shuffle()


Net.print()
print(Icor)
Met.print()

# --- Matching

M = matching(Net, Met, nIter=10, verbose=True)

# Correct matches
print(M)
print(np.count_nonzero([Icor[m[1]]==m[0] for m in M])/n)