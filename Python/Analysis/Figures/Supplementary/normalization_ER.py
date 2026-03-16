import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import project

os.system('clear')

# === Parameters ===========================================================

l_directed = [False, True]
l_scale = ['lin', 'log']

nA = 100
nRun = 100

lw = 2
markersize = 16
fontsize = 12

color = {'directed': '#318CE7', 'undirected': '#FF8C00'}

# ==========================================================================
plt.rcParams.update({'font.size': fontsize})
fig, ax = plt.subplots(1,2, figsize=(10,5))

for i, scale in enumerate(l_scale):

  for directed in l_directed:

    ds = 'directed' if directed else 'undirected'

    fname = project.root + f'/Files/Normalization/ER/{ds}_{scale}_n={nA:d}_nRun={nRun:d}.csv'

    # Load data
    F = pd.read_csv(fname, index_col=0)

    # x-values
    x = np.array([float(i) for i in list(F)])

    # Compute mean and std
    mv = F.mean()
    s = F.std()

    # Simple normalization factor
    f0 = np.minimum(4*x**2, 4*(nA-x)**2) + 1

    # === Display ==========================================================

    ax[i].fill_between(x, mv-s, mv+s, alpha=0.4, facecolor=color[ds])

    ax[i].scatter(x, mv, s=markersize, color=color[ds], label=ds)

    ax[i].set_xlabel('average degree')
    ax[i].set_ylabel('normalization coefficient')

  ax[i].plot(x, f0, '--', color='k', linewidth=lw, zorder=-1)

  # --- Misc display settings

  if scale=='log':
    ax[i].set_xscale('log')
  ax[i].set_yscale('log')

  ax[i].set_ylim(0.5, 1.2e4)

  ax[i].legend()

  ax[i].set_box_aspect(1)

plt.show()