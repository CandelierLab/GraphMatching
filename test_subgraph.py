import os
from Network import *
from  Comparison import *

os.system('clear')

Net = Network(100)
Net.set_rand_edges('ER', 0.1)

Sub, Isim = Net.subnet(100)

M = matching(Net, Sub, nIter=10, verbose=True)

# Correct matches
print(np.count_nonzero([Isim[m[1]]==m[0] for m in M])/Sub.nNd)