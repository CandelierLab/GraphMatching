import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import project

# os.system('clear')

# === Parameters ===========================================================

nA = 20
l_p = np.linspace(0,1,101)
nRun = 10000

# ==========================================================================

Z = None

for i, p in enumerate(l_p):

  fname = project.root + '/Files/Success ratios/ER_p={:.02f}_nA={:d}_nRun={:d}.csv'.format(p, nA, nRun)

  if os.path.exists(fname):

    # Load data
    gamma = pd.read_csv(fname, index_col=0)

    # x-values
    Nsub = np.array([int(i) for i in list(gamma)])
    rho = Nsub/nA

    # Compute mean and std
    m = gamma.mean()
    s = gamma.std()

  else:

    m = np.zeros(len(rho))
    s = np.zeros(len(rho))

  if Z is None:
    Z = np.empty((len(l_p), len(rho)))
  
  Z[i,:] = m

# --- Display --------------------------------------------------------------

plt.style.use('dark_background')

fig, ax = plt.subplots()

S = np.zeros(len(l_p))

for k in range(len(rho)):

  ax.plot(l_p, Z[:,k], '.-')

  S += Z[:,k]/len(l_p)

p_star = l_p[np.argmax(S)]

print(p_star)

ax.axvline(p_star, color='w', linestyle='--')

ax.set_xlabel(r'p')
ax.set_xlim(0, 1)
ax.set_ylabel(r'$\gamma$')
# ax.set_ylim(0, 1)
ax.grid(True)
ax.set_title(rf'$n_A = {nA:d}, p^*={p_star:.02f}$')

ax.set_yscale('log')

plt.show()