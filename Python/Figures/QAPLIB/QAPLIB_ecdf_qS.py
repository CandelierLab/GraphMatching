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

lw = 2
fontsize = 18

# Colors
c = {'2opt': '#CC4F1B', 'e2opt': '#FF9848',
     'FAQ': '#FFA500', 'eFAQ': '#FACC2E',
     'Zager': '#1B2ACC', 'eZager': '#089FFF',
     'GASM': '#3F7F4C', 'eGASM':'#7EFF99',
     'GASM_GPU': '#000000', 'eGASM_GPU':'#000000'}

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

# --- Computation ----------------------------------------------------------

# --- FAQ

s_FAQ = data.q_FAQ
# g_FAQ = data.g_FAQ
# q_FAQ = data.q_FAQ
t_FAQ = data.t_FAQ

# --- 2opt

s_2opt = data.q_2opt
# g_2opt = data.g_2opt
# q_2opt = data.q_2opt
t_2opt = data.t_2opt

# --- Zager

s_Zager = data.q_Zager
# g_Zager = data.g_Zager
# q_Zager = data.q_Zager
t_Zager = data.t_Zager

# --- GASM CPU

s_GASM = data.q_GASM_CPU
# g_GASM = data.g_GASM_CPU
# q_GASM = data.q_GASM_CPU
t_GASM_CPU = data.t_GASM_CPU

# --- GASM GPU

t_GASM_GPU = data.t_GASM_GPU

# NaN handling
# s_FAQ[np.isnan(s_FAQ)] = 1
# s_2opt[np.isnan(s_2opt)] = 1
# s_Zager[np.isnan(s_Zager)] = 1
# s_GASM[np.isnan(s_GASM)] = 1

# --- Display --------------------------------------------------------------

plt.rcParams.update({'font.size': fontsize})
fig, ax = plt.subplots(1,2, figsize=(10,5))

# --- Score ratios

cdf_FAQ = ecdf(s_FAQ)
cdf_2opt = ecdf(s_2opt)
cdf_Zager = ecdf(s_Zager)
cdf_GASM = ecdf(s_GASM)

cdf_FAQ.cdf.plot(ax[0], color=c['FAQ'], linewidth=lw, label='FAQ')
# cdf_2opt.cdf.plot(ax[0], color=c['2opt'], linewidth=lw, label='2opt')
# cdf_Zager.cdf.plot(ax[0], color=c['Zager'], linewidth=lw, label='Zager')
cdf_GASM.cdf.plot(ax[0], color=c['GASM'], linewidth=lw, label='GASM')

cdf_FAQ = ecdf(t_FAQ)
cdf_2opt = ecdf(t_2opt)
cdf_Zager = ecdf(t_Zager)
cdf_GASM_CPU = ecdf(t_GASM_CPU)
cdf_GASM_GPU = ecdf(t_GASM_GPU)

cdf_FAQ.cdf.plot(ax[1], color=c['FAQ'], linewidth=lw, label='FAQ')
cdf_2opt.cdf.plot(ax[1], color=c['2opt'], linewidth=lw, label='2opt')
cdf_Zager.cdf.plot(ax[1], color=c['Zager'], linewidth=lw, label='Zager')
cdf_GASM_CPU.cdf.plot(ax[1], color=c['GASM'], linewidth=lw, label='GASM')
cdf_GASM_GPU.cdf.plot(ax[1], color=c['GASM_GPU'], linewidth=lw, label='GASM GPU')

ax[0].legend()
# ax[0].set_xscale('log')
# ax[0].set_xlim(0.8, 100)
ax[0].set_ylim(0, 1)
ax[0].set_box_aspect(1)

ax[0].set_xlabel('$q_S$')
ax[0].set_ylabel('ecdf')

ax[1].legend()
ax[1].set_xscale('log')
# ax[1].set_xlim(0.8, 100)
ax[1].set_ylim(0, 1)
ax[1].set_box_aspect(1)

ax[1].set_xlabel('$t$ (ms)')
ax[1].set_ylabel('ecdf')

#  --- Output --------------------------------------------------------------

if figfile is None:

  plt.show()

else:

  print(f'Saving file {figfile} ...', end='', flush=True)
  tref = time.time()

  plt.savefig(project.root + '/Figures/' + figfile)

  print(' {:.02f} sec'.format((time.time() - tref)))