import os
import math
import numpy as np
import matplotlib.pyplot as plt

os.system('clear')

# --------------------------------------------------------------------------

plt.style.use('dark_background')
fig, ax = plt.subplots()

x = np.arange(0,10)
y = [math.factorial(2*n)/math.factorial(n)**2 for n in x]

ax.plot(x, y, '.-')

# ax.set_xscale('log')
ax.set_yscale('log')

ax.grid(True)

plt.show()