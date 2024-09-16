'''
-- Display --

QAPLIB: scores, gamma and q

'''

import os
import argparse
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

import project

# === Parameters ===========================================================

l_algo = ['FAQ', '2opt', 'Zager', 'GASM']

directed = False
nA = 20

# --------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename', help='File to save the figure')
args = parser.parse_args()
figfile = args.filename

# ==========================================================================

if figfile is None:
  os.system('clear')

# --- Load data ------------------------------------------------------------

datapath = project.root + f'/Files/QAPLIB/sgqt.csv'

if os.path.exists(datapath):

  # Load data
  data = pd.read_csv(datapath)

print(data)

# --- Computation ----------------------------------------------------------

# --- FAQ

s_FAQ = np.sort(data.s_FAQ/data.s_sol)
g_FAQ = np.flip(np.sort(data.g_FAQ))
q_FAQ = np.flip(np.sort(data.q_FAQ))

# --- 2opt

s_2opt = np.sort(data.s_2opt/data.s_sol)
g_2opt = np.flip(np.sort(data.g_2opt))
q_2opt = np.flip(np.sort(data.q_2opt))


# --- Zager

s_Zager = np.sort(data.s_Zager/data.s_sol)
g_Zager = np.flip(np.sort(data.g_Zager))
q_Zager = np.flip(np.sort(data.q_Zager))

# --- GASM

s_GASM = np.sort(data.s_GASM_CPU/data.s_sol)
g_GASM = np.flip(np.sort(data.g_GASM_CPU))
q_GASM = np.flip(np.sort(data.q_GASM_CPU))

# NaN handling
s_FAQ[np.isnan(s_FAQ)] = 1
s_2opt[np.isnan(s_2opt)] = 1
s_Zager[np.isnan(s_Zager)] = 1
s_GASM[np.isnan(s_GASM)] = 1

# --- Display --------------------------------------------------------------

plt.style.use('dark_background')
fig, ax = plt.subplots(1, 1, figsize=(5,5))

plt.scatter(s_GASM, s_FAQ, label='FAQ')
plt.scatter(s_GASM, s_2opt, label='2opt')
plt.scatter(s_GASM, s_Zager, label='Zager')

range = [0, 60]
plt.plot(range, range, linestyle='--', color='w')

ax.legend()
ax.set_xscale('log')
ax.set_yscale('log')

#  --- Output --------------------------------------------------------------

if figfile is None:

  plt.show()

else:

  print(f'Saving file {figfile} ...', end='', flush=True)
  tref = time.time()

  plt.savefig(project.root + '/Figures/' + figfile)

  print(' {:.02f} sec'.format((time.time() - tref)))