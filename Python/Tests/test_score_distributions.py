import os
import matplotlib.pyplot as plt

import project
from Network import *
from  Comparison import *

os.system('clear')

# --- Parameters -----------------------------------------------------------

nA = 50
pA = 0.1

nB = 10
pB = 0.9

nIter = 30

f = None

# --------------------------------------------------------------------------

# --- Networks

NetA = Network(nA)
NetA.set_rand_edges('ER', pA)

NetB = Network(nB)
NetB.set_rand_edges('ER', pB)

# --- Figure

plt.style.use('dark_background')
fig, ax = plt.subplots()

# --- Computation

# After normalisation
# mean(x) = 1
# mean(x-x0) = 0 

s = []
sd = []

X0 = np.ones((nA, nB))

for iter in range(1, nIter):

  # Structure & attribute scores
  X = scores(NetA, NetB, nIter=iter, normalization=f)[0]

  X = X/np.mean(X)
  s.append(np.std(X))
  sd.append(np.std(X-X0))
  # dmx.append(np.abs(np.mean(X)-np.mean(X0)))
  # mdx.append(np.mean(np.abs(X-X0)))

  # Update 
  X0 = X

  # kde = stats.gaussian_kde(X.flatten()/np.mean(X))
  # x = np.linspace(0, 3, 100)
  # p = kde(x)

  # ax.plot(x, p, label=nIter)

# s = s[-1]-s
# s /= s[0]
# sd /= sd[0] 

ax.plot(s, '.-', label='std')
ax.plot(sd, '.-', label='std $\delta$')

# ax.set_yscale('log')
ax.legend()
ax.grid(True)

plt.show()
