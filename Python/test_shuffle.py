import os
from Network import *
from Comparison import *

os.system('clear')

# --- Parameters -----------------------------------------------------------

n = 20
p = 0.5 #round(n/10)

nIter = 10

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

start = time.time()

X = scores(Net, Met, nIter=nIter)[0]

print((time.time() - start)*1000)
print('Check: ', X[0][0])

print('')

start = time.time()

X = scores_cpp(Net, Met, nIter=nIter)[0]

print((time.time() - start)*1000)
print('Check: ', X[0][0])

# M = matching(Net, Met, nIter=nIter, verbose=True)

# # Correct matches
# print(np.count_nonzero([Icor[m[1]]==m[0] for m in M])/n)