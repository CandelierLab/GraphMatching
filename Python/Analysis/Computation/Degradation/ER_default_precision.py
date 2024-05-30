'''
Default precision
'''

import os
import numpy as np
import pandas as pd
import time

import project
from Graph import *
from Comparison import *

os.system('clear')

# === Parameters ===========================================================

directed = False

nA = 200
p = np.log(nA)/nA

l_delta = np.linspace(0, 1, 21)
l_precision = [0.01]

# ==========================================================================

Ga = Gnp(nA, p, directed=directed)

# Edge atrribute
Ga.add_edge_attr('gauss')

wA = Ga.edge_attr[0]['values']

rho_a = np.zeros(l_delta.size)

for i, delta in enumerate(l_delta):
  
  # Degradation: remove edges
  Gb, gt = Ga.degrade('ed_rm', delta)

  wB = Gb.edge_attr[0]['values']

  # Edge weights differences
  W = np.subtract.outer(wA, wB)

  rho_a[i] = np.std(W)
  
print(rho_a)