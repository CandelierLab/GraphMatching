import os
import Network
from  Comparison import *

os.system('clear')

Net = Network.Random(30, 0.5)

Sub, Idx = Net.subnet(10)

# print(Net.Adj)
# print(Idx)
# print(Sub.Adj)

M = matching(Net, Sub, nIter=10, verbose=True)

# Correct matches
print(np.count_nonzero([Idx[m[1]]==m[0] for m in M])/Sub.nNd)