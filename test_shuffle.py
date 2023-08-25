import os
from Network import *
from  Comparison import *

os.system('clear')

# --- Parameters -----------------------------------------------------------

n = 10
p = 0.5 #round(n/10)

# n = 1000
# p = round(n/10)

# --------------------------------------------------------------------------

Net = Network(n)
Net.set_rand_edges('ER', p)

# --- Identity

Met = copy.deepcopy(Net)
Icor = np.arange(n)

# --- Shuffling

Met, Icor = Net.shuffle()


# Net.print()
# print(Icor)
# Met.print()

# --- Matching

X, Y = scores(Net, Met, nIter=100)
with np.printoptions(precision=3, suppress=True):
  print(X)
  print(np.mean(X/np.sqrt(np.sum(X**2))))


# M = matching(Net, Met, nIter=10, verbose=True)

# # Correct matches
# print(np.count_nonzero([Icor[m[1]]==m[0] for m in M])/n)