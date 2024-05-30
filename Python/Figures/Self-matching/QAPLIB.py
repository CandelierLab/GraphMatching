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

lw = 3
fontsize = 24

# Colors
c = {'2opt': '#CC4F1B', 'e2opt': '#FF9848',
     'FAQ': '#FFA500', 'eFAQ': '#FACC2E',
     'Zager': '#1B2ACC', 'eZager': '#089FFF',
     'GASM': '#3F7F4C', 'eGASM':'#7EFF99'}

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

plt.rcParams.update({'font.size': fontsize})
fig, ax = plt.subplots(1,1, figsize=(10,10))

# --- Score ratios

cdf_FAQ = ecdf(s_FAQ)
cdf_2opt = ecdf(s_2opt)
cdf_Zager = ecdf(s_Zager)
cdf_GASM = ecdf(s_GASM)
cdf_FAQ.cdf.plot(ax, color=c['FAQ'], linewidth=lw, label='FAQ')
cdf_2opt.cdf.plot(ax, color=c['2opt'], linewidth=lw, label='2opt')
cdf_Zager.cdf.plot(ax, color=c['Zager'], linewidth=lw, label='Zager')
cdf_GASM.cdf.plot(ax, color=c['GASM'], linewidth=lw, label='GASM')

ax.legend()
ax.set_xscale('log')
ax.set_xlim(0.8, 100)
ax.set_ylim(0, 1)
ax.set_box_aspect(1)

ax.set_xlabel('ratio')
ax.set_ylabel('ecdf')

#  --- Output --------------------------------------------------------------

if figfile is None:

  plt.show()

else:

  print(f'Saving file {figfile} ...', end='', flush=True)
  tref = time.time()

  plt.savefig(project.root + '/Figures/' + figfile)

  print(' {:.02f} sec'.format((time.time() - tref)))