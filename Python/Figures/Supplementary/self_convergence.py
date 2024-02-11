import os
import time
import matplotlib.pyplot as plt

import project
from Network import *
from Comparison import *

os.system('clear')

# === Parameters ===========================================================

# n = 1000
n = 30

# Average number of edges per node
l_nepn = [3] #[0.25, 0.5, 0.75, 1, 2, 3]

nIterMax = 1

# === Functions ============================================================

def probe(V, param, out):

  # Local variables
  X = V['X']

  # Other parameters
  n = param['NetA'].nNd 
  Icor = param['Icor']

  # Matching
  M = matching(param['NetA'], param['NetB'], scores=X)

  # print(M)

  # Accuracy
  # M.compute_accuracy(Icor)

  # Output
  # out.append(M.accuracy)
  out.append(0)


# ==========================================================================

l_gamma = []

for nepn in l_nepn:

  print('Nepn {:.01f} ...'.format(nepn), end='')
  start = time.time()

  # --- Network

  Net = Network(n)
  Net.set_rand_edges('ER', n_epn=nepn)

  print(Net)

  Set, Icor = Net.shuffle()

  # --- Convergence

  # Scores
  X, Y, gamma = compute_scores(Net, Set, nIter=nIterMax, i_function=probe, initial_evaluation=True, i_param={'Icor': Icor})
    
  l_gamma.append(gamma)

  print('{:.02f} sec'.format((time.time() - start)))

# === Display =================================================================

# plt.style.use('dark_background')

# fig, ax = plt.subplots()

# for i, gamma in enumerate(l_gamma):

#   ax.plot(gamma, '.-', label=l_nepn[i])

# ax.set_xlabel('Iteration')
# ax.set_ylabel(r'Accuracy $\gamma$')
# ax.set_title('legend: average number of edge per node')

# # ax.set_xlim(0, max(l_nIter))
# ax.xaxis.grid(True)
# ax.set_ylim(0,1)
# ax.legend()

# plt.show()