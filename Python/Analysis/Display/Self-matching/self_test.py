import os
import numpy as np
import pandas as pd
import time

import project
from Network import *
from Comparison import *

os.system('clear')

# === Parameters ===========================================================

nA = 10
p = 0.15

# ==========================================================================

NetA = Network(nA)
NetA.set_rand_edges('ER', p_edges=p)

NetB, Idx = NetA.shuffle()

C = Comparison(NetA, NetB, algorithm='GASM', eta=0.001)

M = C.get_matching()
M.compute_accuracy(Idx)

# --- Output

pa.line(os.path.basename(__file__))
print()

# pa.matrix(C.X, title='Node scores')
# pa.matrix(C.Y)

print(M)

print(Idx)

import matplotlib.pyplot as plt
nx.draw(NetA.G)
plt.show()  # pyplot draw()
