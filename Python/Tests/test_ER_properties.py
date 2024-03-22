import os
import numpy as np
from scipy.stats.kde import gaussian_kde
import matplotlib.pyplot as plt

import project
from Network import *
from  Comparison import *

import paprint as pa

os.system('clear')

# === Parameters ===========================================================

nA = 200
# p = 0.9

algo = 'GASM'

# np.random.seed(0)

# --------------------------------------------------------------------------

p = np.log(nA)/nA

# ==========================================================================

# --- Random graphs

Ga = Network(nA)
Ga.set_rand_edges('ER', p_edges=p)

# Get in and out degree
di = np.sum(Ga.Adj, axis=0)
do = np.sum(Ga.Adj, axis=1)
d = np.concatenate((di, do))

# KDE
kde = gaussian_kde(d)


plt.style.use('dark_background')
fig, ax = plt.subplots()

x = np.linspace(0,20,101)
ax.plot(x, kde(x))

plt.show()