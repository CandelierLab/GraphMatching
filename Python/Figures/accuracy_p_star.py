import os
import numpy as np
import re
import matplotlib.pyplot as plt

import project

os.system('clear')

# === Parameters ===========================================================

dname = project.root + '/Files/Success ratios/p_star/'

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

# === Fit ==================================================================

P = np.polyfit(np.log(nA), np.log(p_star), 1)
a = P[0]
b = np.exp(P[1])

# a = -1
# b = 2

x = np.geomspace(10, 100, 100)
y = b*x**a
 
# === Display ==============================================================

fig, ax = plt.subplots()

ax.plot(x, y, color='r')
ax.plot(nA, p_star, '+')

ax.set_xscale('log')
ax.set_yscale('log')

ax.set_xlabel(r'$n_A$')
ax.set_ylabel(r'$p^\ast$')

ax.set_title(f'a={a:.2f}, b={b:.2f}')

plt.show()