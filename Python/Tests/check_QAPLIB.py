import os
from scipy.optimize import quadratic_assignment
import matplotlib.pyplot as plt

import project
from Graph import *
from  Comparison import *
from QAPLIB import QAPLIB

import paprint as pa

os.system('clear')

# === Parameters ===========================================================

# id = 'esc128'

# ==========================================================================

Q = QAPLIB()

for id in Q.l_inst:

  I = Q.get(id)

  S = np.trace(I.A.T @ I.B[I.s, :][:, I.s])

  if I.score!=S:
    print(id, I.score, S)
