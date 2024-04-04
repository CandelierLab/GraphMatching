import os
import argparse
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

import project

os.system('clear')

# === Parameters ===========================================================

nA = 20
nRun = 1000

# --------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename', help='File to save the figure')
args = parser.parse_args()
figfile = args.filename

# --------------------------------------------------------------------------

datapath = project.root + f'/Files/Self-matching/ER/nA={nA:d}_nRun={nRun:d}.csv'

# ==========================================================================

if os.path.exists(datapath):

  # Load data
  df = pd.read_csv(datapath)

  # Retrieve l_p and l_eta

  l_p = np.unique(df.p)
  l_eta = np.unique(df.eta)

l_eta = np.array([l_eta[2]])

# --- Display --------------------------------------------------------------

plt.style.use('dark_background')
fig, ax = plt.subplots(1,2)

# Colors
cm = plt.cm.gist_rainbow(np.linspace(0, 1, l_eta.size))

g_FAQ = np.zeros(l_p.size)
q_FAQ = np.zeros(l_p.size)

g_Zager = np.zeros(l_p.size)
q_Zager = np.zeros(l_p.size)

for i, eta in enumerate(l_eta):

  data = df.loc[df['eta'] == eta]

  # Accuracy
  g_FAQ += data.g_FAQ.to_list()
  g_Zager += data.g_Zager.to_list()
  ax[0].plot(data.p, data.g_GASM, '-', color=cm[i], label=f'$\eta = {eta:g}$')

  # Structural quality
  q_FAQ += data.q_FAQ.to_list()
  q_Zager += data.q_Zager.to_list()
  ax[1].plot(data.p, data.q_GASM, '-', color=cm[i], label=f'$\eta = {eta:g}$')

ax[0].plot(l_p, g_FAQ/l_eta.size, '-', color='c', label='FAQ')
ax[1].plot(l_p, q_FAQ/l_eta.size, '-', color='c', label='FAQ')

ax[0].plot(l_p, g_Zager/l_eta.size, '--', color='w', label='Zager')
ax[1].plot(l_p, q_Zager/l_eta.size, '--', color='w', label='Zager')

ax[0].set_ylim([0, 1])
ax[1].set_ylim([0.95, 1.001])

ax[0].set_xlabel('p')
ax[1].set_xlabel('p')

ax[0].set_ylabel('$\gamma$')
ax[1].set_ylabel('$q$')

ax[0].legend()
ax[1].legend()

ax[0].grid(True)

#  --- Output --------------------------------------------------------------

if figfile is None:
  
  plt.show()

else:

  print(f'Saving file {figfile} ...', end='', flush=True)
  tref = time.time()

  plt.savefig(project.root + '/Figures/' + figfile)

  print(' {:.02f} sec'.format((time.time() - tref)))