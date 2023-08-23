import os
from Network import *
from  Comparison import *

os.system('clear')

Net = Network(10)
Net.set_rand_edges('ER', 0.1)

Det, Idx = Net.degrade('struct', p=0.5)

M = matching(Net, Det, nIter=10, verbose=True)

print(M)

# Correct matches
print(np.count_nonzero([m[1]==m[0] for m in M])/Net.nNd)