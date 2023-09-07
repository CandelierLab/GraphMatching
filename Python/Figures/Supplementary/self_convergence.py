import os
import numpy as np
import time
import matplotlib.pyplot as plt

import project
from Network import *
from Comparison import *

os.system('clear')

# === Parameters ===========================================================

n = 1000

# Average number of edges per node
l_nepn = [0.25, 0.5, 0.75, 1, 2, 3]

nIter = 10

# === Functions ============================================================

def probe(V, param):
  
  # Local variables
  X = V['X']

  # Other parameters
  n = param['n']
  Icorr = param['Icorr']

  # Matching
  I, J = linear_sum_assignment(X, True)
  M = [(I[k], J[k]) for k in range(len(I))]

  # Correct matches
  rho = np.count_nonzero([Icorr[J[k]]==I[k] for k in range(len(I))])/n

  # Output
  return rho

# ==========================================================================

l_rho = []

for nepn in l_nepn:

  print('Nepn {:.01f} ...'.format(nepn), end='')
  start = time.time()

  # --- Network

  Net = Network(n)
  Net.set_rand_edges('ER', int(nepn*n))

  Set, Icorr = Net.shuffle()

  # --- Convergence

  # Scores
  X, Y, rho = scores(Net, Set, nIter=nIter, i_function=probe, initial_evaluation=True,
                     i_param={'n': n, 'Icorr': Icorr})
    
  l_rho.append(rho)

  print('{:.02f} sec'.format((time.time() - start)))

# === Display =================================================================

# plt.style.use('dark_background')

fig, ax = plt.subplots()

for i, rho in enumerate(l_rho):

  ax.plot(rho, '.-', label=l_nepn[i])

ax.set_xlabel('Iteration')
ax.set_ylabel(r'Ratio of correct matches $\rho$')
ax.set_title('legend: average number of edge per node')

# ax.set_xlim(0, max(l_nIter))
ax.set_ylim(0,1)
ax.legend()

plt.show()