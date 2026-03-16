import os
import numpy as np
import re
import matplotlib.pyplot as plt

import project

os.system('clear')

# === Parameters ===========================================================

dname = project.root + '/Files/Subgraph/p_star/'

# --- Plot parameters

err_alpha = 0.2
lw = 3
fontsize = 16

# ==========================================================================

# --- Load data

# Initialize arrays
nA = []
p_star = []

# Regular expression
p = re.compile("nA=(.*)\.txt")

for fname in os.listdir(dname):

  with open(dname + fname) as f:
    nA.append(int(p.search(fname).group(1)))
    p_star.append(float(f.read()[0:-1]))
    
# Sort it
nA, p_star = zip(*sorted(zip(nA, p_star)))

x = np.geomspace(10, 1000, 100)
y = 2/x
 
# === Display ==============================================================

plt.rcParams.update({'font.size': fontsize})
fig, ax = plt.subplots(figsize=(5,5))

ax.plot(x, y, color='r')
ax.plot(nA, p_star, '+', color='k')

ax.set_xscale('log')
ax.set_yscale('log')

ax.set_xlabel(r'$n_A$')
ax.set_ylabel(r'$p^\ast$')

plt.show()