import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import project

os.system('clear')

# === Parameters ===========================================================

nRun = 10

# --------------------------------------------------------------------------

fname = project.root + f'/Files/Speed/ER_nRun={nRun:d}.csv'

# ==========================================================================

if os.path.exists(fname):

  # Load data
  df = pd.read_csv(fname)

  # Retrieve l_n
  l_n = np.unique(df.n)

# --- Display --------------------------------------------------------------

plt.style.use('dark_background')
fig, ax = plt.subplots()

ax.plot(l_n, df.FAQ, '.-', label='FAQ')
ax.plot(l_n, df.Zager, '.-', label='Zager')
ax.plot(l_n, df.GASM, '.-', label='GASM')
ax.plot(l_n, df.GASM_LAP, '--', label='GASM_LAP')

ax.set_xscale('log')
ax.set_yscale('log')

ax.legend()

plt.show()