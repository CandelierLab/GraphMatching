import os
import numpy as np
import time
import matplotlib.pyplot as plt

import project
from Graph import *
from Comparison import *

os.system('clear')

# === Parameters ===========================================================

n = 100

# Average number of edges per node
# l_nepn = [0.1, 1, 10, 50]
l_nepn = np.geomspace(0.1, 100, 20)

nIter = 10

# === Functions ============================================================

def probe(V, param, out):

  f = np.mean(V['X'])

  # Output
  out.append(f)

# ==========================================================================

l_f0 = []
l_f1 = []
l_ff = []

for nepn in l_nepn:

  print('Nepn {:.01f} ...'.format(nepn), end='')
  start = time.time()

  # --- Network

  NetA = Network(n)
  NetA.set_rand_edges('ER', int(nepn*n))

  NetB, Icorr = NetA.shuffle()

  # NetB = Network(int(n))
  # NetB.set_rand_edges('ER', int(nepn*n))

  # --- Convergence

  # Scores
  X, Y, output = scores(NetA, NetB, nIter=nIter, normalization=1,
                        i_function=probe, initial_evaluation=True)
    
  # g = [output[i]/output[i-1] for i in range(1,len(output))]
  # g.insert(0,1)
  # l_f.append(np.array(g))

  l_f0.append(4*NetA.nEd*NetB.nEd/NetA.nNd/NetB.nNd)
  l_f1.append(4*((NetA.nEd*NetB.nEd/NetA.nNd/NetB.nNd)**0.9)+5)
  l_ff.append(output[-1]/output[-2])

  print('{:.02f} sec'.format((time.time() - start)))

# === Display =================================================================

plt.style.use('dark_background')

fig, ax = plt.subplots()

# ax.axhline(1, linestyle='--', color='w')

# for i, f in enumerate(l_ff):
#   ax.plot(f, '.-', label=l_nepn[i])

ax.plot(l_nepn/n, l_f0, '--', color='w')
ax.plot(l_nepn/n, l_f1, '-', color='r')
ax.plot(l_nepn/n, l_ff, '.-')

ax.set_xlabel('Ratio of edge per node')
ax.set_ylabel('Normalization factor')

ax.set_xscale('log')
ax.set_yscale('log')

# ax.legend()
# ax.set_title('legend: average number of edge per node')

# ax.grid(True)

plt.show()