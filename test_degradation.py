import os
from Network import *
from  Comparison import *

os.system('clear')

Net = Network(5)
Net.set_rand_edges('ER', 5)

Det, Isim = Net.degrade('struct', n=1)

Net.print()
print(Isim, np.setdiff1d(range(Net.nEd), Isim))
Det.print()

X, Y = scores(Net, Det)

with np.printoptions(precision=2, suppress=True):
  print(X)
# print(Y)

M = matching(Net, Det, nIter=100, verbose=True)

print(M)

# # Correct matches
# print(np.count_nonzero([m[1]==m[0] for m in M])/Net.nNd)