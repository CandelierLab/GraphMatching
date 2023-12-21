import os
import numpy as np
import matplotlib.pyplot as plt

os.system('clear')

# --------------------------------------------------------------------------

plt.style.use('dark_background')
fig, ax = plt.subplots()

A = np.arange(10,100,11)

for a in A:

  x = np.linspace(0, a, 100)
  y = (a-x)**x

  ax.plot(x, y, '.-')

ax.set_yscale('log')

plt.show()