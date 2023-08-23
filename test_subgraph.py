import os
from Network import *
from  Comparison import *

os.system('clear')

Net = Network(30)
Net.set_rand_edges('ER', 0.1)

Sub, Idx = Net.subnet(10)

M = matching(Net, Sub, nIter=10, verbose=True)

# Correct matches
print(np.count_nonzero([Idx[m[1]]==m[0] for m in M])/Sub.nNd)