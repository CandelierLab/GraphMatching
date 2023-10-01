'''
Searching for p*
'''

import os
import numpy as np
import pandas as pd
import time

import project
from Network import *
from Comparison import *
import paprint as pa

import matplotlib.pyplot as plt

# os.system('clear')

# === Parameters ===========================================================

nA = 10
rho = 0.75
nRun = 100000

sigma = 2/nA

dname = project.root + '/Files/Success ratios/p_star/'

force = True

# ==========================================================================

fname = dname + f'nA={nA:d}.txt'

if force or not os.path.exists(fname):

  print('Searching p_star ...')

  n = int(np.round(rho*nA))

  p = np.random.rand(nRun)*4/nA
  g = np.empty(nRun)

  x = np.linspace(0, 4/nA, 10000)
  y = np.zeros(10000)

  for i in range(nRun):

    Net = Network(nA)
    Net.set_rand_edges('ER', p[i])

    Sub, Idx = Net.subnet(n)
    M = matching(Net, Sub)

    # Correct matches
    g[i] = np.count_nonzero([Idx[m[1]]==m[0] for m in M])/n

    y += g[i]*np.exp(-(((x-p[i])/sigma)**2)/2)/nRun

    if not i % (nRun/100):
      print(f'{int(i*100/nRun):02d} %')

  p_star = x[np.argmax(y)]

  print('p_star: ', p_star)

  # --- Save

  with open(fname, 'w') as f:
    f.write(str(p_star))