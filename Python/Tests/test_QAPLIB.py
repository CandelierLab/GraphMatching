import os

import project
from Graph import *
from  Comparison import *

import paprint as pa

os.system('clear')

# === Parameters ===========================================================

n = 0

algo = 'FAQ'

# ==========================================================================

Ga, Gb, gt = qaplib(n)

Ga.print()
Gb.print()
print(gt)

C = Comparison(Ga, Gb, verbose=True)
M = C.get_matching(algorithm=algo)
M.compute_accuracy(gt)

print(M)
