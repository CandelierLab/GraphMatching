'''
Golden search algorithm for p*
'''

import os
import numpy as np
import pandas as pd
import time

import project
from Network import *
from Comparison import *
import paprint as pa

os.system('clear')

# === Parameters ===========================================================

l_nA = [10]
rho = [0.9]
nRun = 100000

tol = 1e-3

dname = project.root + '/Files/Success ratios/p_star/'

force = True

# ==========================================================================

invphi = (np.sqrt(5) - 1) / 2  # 1 / phi
invphi2 = (3 - np.sqrt(5)) / 2  # 1 / phi^2

def gsearch(f, a, b, nA, tol=1e-5, h=None, c=None, d=None, fc=None, fd=None):
    """Golden-section search, recursive.

    Given a function f with a single local minimum in
    the interval [a,b], gss returns a subset interval
    [c,d] that contains the minimum with d-c <= tol.

    Example:
    >>> f = lambda x: (x-2)**2
    >>> a = 1
    >>> b = 5
    >>> tol = 1e-5
    >>> (c,d) = gssrec(f, a, b, tol)
    >>> print (c, d)
    1.9999959837979107 2.0000050911830893
    """

    (a, b) = (min(a, b), max(a, b))
    if h is None:
        h = b - a
    if h <= tol:
        return (a, b)
    if c is None:
        c = a + invphi2 * h
    if d is None:
        d = a + invphi * h
    if fc is None:
        fc = f(c, nA)
    if fd is None:
        fd = f(d, nA)
    if fc > fd:  # fc < fd to find the minimum
        return gsearch(f, a, d, nA, tol, h*invphi, c=None, fc=None, d=c, fd=fc)
    else:
        return gsearch(f, c, b, nA, tol, h*invphi, c=d, fc=fd, d=None, fd=None)

def evaluate(p, nA):
    
  print(f'Evaluate p={p:.06f} ...', end='')
  start = time.time()

  gamma = 0

  for r in rho:

    n = int(np.round(r*nA))

    g = np.empty(nRun)

    for i in range(nRun):

      Net = Network(nA)
      Net.set_rand_edges('ER', p)

      Sub, Idx = Net.subnet(n)
      M = matching(Net, Sub)

      # Correct matches
      g[i] = np.count_nonzero([Idx[m[1]]==m[0] for m in M])/n

    gamma += np.mean(g)

  print(' {:.02f} sec'.format((time.time() - start)))

  return gamma

# --- Main loop ------------------------------------------------------------

for nA in l_nA:

  fname = dname + f'nA={nA:d}.txt'

  if os.path.exists(fname) and not force:
     continue

  pa.line(text=f'nA = {nA:d}')

  a,b = gsearch(evaluate, 0, 0.2, nA, tol=tol)

  p_star = (a+b)/2

  print('p_star: ', p_star)

  # --- Save
  with open(fname, 'w') as f:
    f.write(str(p_star))

