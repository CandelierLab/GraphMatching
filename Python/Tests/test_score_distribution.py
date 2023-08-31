import os
import numpy as np
import itertools
import matplotlib.pyplot as plt

os.system('clear')

# === Parameters ===========================================================

N = 8

# ==========================================================================

# --- Sample matrix

M = np.random.rand(N,N)

# --- Solutions

Sol = list(itertools.permutations(np.arange(N)))

# --- Solutions matching scores

I = np.arange(N)
s = []
for J in Sol:
  s.append(np.sum(M[I,J]))

# --- Display

plt.style.use('dark_background')

fig, ax = plt.subplots()

ax.hist(s, bins=100)

plt.show()