import os
from scipy import stats
import matplotlib.pyplot as plt

import project
from Network import *
from  Comparison import *

os.system('clear')

# --- Parameters -----------------------------------------------------------

nA = 50
pA = 0.5

nB = 50
pB = 0.5

# --------------------------------------------------------------------------

NetA = Network(nA)
NetA.set_rand_edges('ER', pA)

# NetB = NetA.shuffle()[0]

NetB = Network(nB)
NetB.set_rand_edges('ER', pB)

plt.style.use('dark_background')
fig, ax = plt.subplots()

for nIter in range(1,10):

  # Structure & attribute scores
  X = scores(NetA, NetB, nIter=nIter)[0]

  kde = stats.gaussian_kde(X.flatten()/np.mean(X))
  x = np.linspace(0, 3, 100)
  p = kde(x)

  ax.plot(x, p, label=nIter)

ax.legend()

plt.show()
