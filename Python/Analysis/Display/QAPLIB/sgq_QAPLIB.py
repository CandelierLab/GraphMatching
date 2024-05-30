'''
-- Display --

QAPLIB: scores, gamma and q

'''

import os
import argparse
import numpy as np
from scipy.stats import ecdf
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

datapath = project.root + f'/Files/QAPLIB/sgq.csv'

if os.path.exists(datapath):

  # Load data
  data = pd.read_csv(datapath)

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

s_GASM = np.sort(data.s_GASM/data.s_sol)
g_GASM = np.flip(np.sort(data.g_GASM))
q_GASM = np.flip(np.sort(data.q_GASM))

# NaN handling
s_FAQ[np.isnan(s_FAQ)] = 1
s_2opt[np.isnan(s_2opt)] = 1
s_Zager[np.isnan(s_Zager)] = 1
s_GASM[np.isnan(s_GASM)] = 1

# --- Display --------------------------------------------------------------

plt.style.use('dark_background')
fig, ax = plt.subplots(1, 2, figsize=(10,5))

# --- Scores

cdf_FAQ = ecdf(s_FAQ)
cdf_2opt = ecdf(s_2opt)
cdf_Zager = ecdf(s_Zager)
cdf_GASM = ecdf(s_GASM)

cdf_FAQ.cdf.plot(ax[0], label='FAQ')
cdf_2opt.cdf.plot(ax[0], label='2opt')
cdf_Zager.cdf.plot(ax[0], label='Zager')
cdf_GASM.cdf.plot(ax[0], label='GASM')

ax[0].legend()
ax[0].set_xscale('log')

# --- Scores

s_FAQ = data.s_FAQ/data.s_sol
s_GASM = data.s_GASM/data.s_sol

ax[1].plot(s_FAQ, label='FAQ')
ax[1].plot(s_GASM, label='GASM')

ax[1].legend()

# # --- Accuracy

# ax[1].plot(g_FAQ, '-', label='FAQ')
# ax[1].plot(g_2opt, '-', label='2opt')
# ax[1].plot(g_Zager, '-', label='Zager')
# ax[1].plot(g_GASM, '-', label='GASM')

# ax[1].legend()
# ax[1].set_ylim(0, 1)

# # --- Structural quality

# ax[2].plot(q_FAQ, '-', label='FAQ')
# ax[2].plot(q_2opt, '-', label='2opt')
# ax[2].plot(q_Zager, '-', label='Zager')
# ax[2].plot(q_GASM, '-', label='GASM')

# ax[2].legend()
# ax[2].set_ylim(0, 1)


#  --- Output --------------------------------------------------------------

if figfile is None:

  plt.show()

else:

  print(f'Saving file {figfile} ...', end='', flush=True)
  tref = time.time()

  plt.savefig(project.root + '/Figures/' + figfile)

  print(' {:.02f} sec'.format((time.time() - tref)))