import os

import project
from Network import *
from Comparison import *

os.system('clear')

# --- Parameters -----------------------------------------------------------

n = 1000
# p = 0.01
p = round(n*10)

nIter = 10

# n = 1000
# p = round(n/10)

# --------------------------------------------------------------------------

Net = Network(n)
Net.set_rand_edges('ER', p)

# --- Identity

# Met = copy.deepcopy(Net)
# Icor = np.arange(n)

# --- Shuffling

Met, Icor = Net.shuffle()

print('Number of operation: ', Net.nEd*Met.nEd)


# --- Matching

start = time.time()

X = scores(Net, Met, nIter=nIter)[0]

print((time.time() - start)*1000)
print('Check: ', X[0,0:5])

print('')

start = time.time()

X = scores_cpp(Net, Met, nIter=nIter)[0]

print((time.time() - start)*1000)
print('Check: ', X[0,0:5])

# M = matching(Net, Met, nIter=nIter, verbose=True)

# # Correct matches
# print(np.count_nonzero([Icor[m[1]]==m[0] for m in M])/n)