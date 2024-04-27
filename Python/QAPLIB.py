'''
QAPLIB wrapper
'''

import os
import numpy as np

import project
from Graph import *

class QAPLIB:

  def __init__(self):
    
    # Paths
    self.root = project.root + '/Files/QAPLIB/'
    self.dinst = self.root + 'Instances/'
    self.dsol = self.root + 'Solutions/'

    # List instances with solutions
    self.l_inst = sorted([s[:-4] for s in os.listdir(self.dsol)])
    self.n_inst = len(self.l_inst)

  def get(self, n):
    '''
    Get n-th instance with a solution
    '''

    if n >= self.n_inst:
      raise Exception(f'Please choose a solution in [0,{self.n_inst-1}].') 
    
    # --- Parse instance ---------------------------------------------------

    f_inst = self.dinst + self.l_inst[n] + '.dat'
    
    with open(f_inst, 'r') as f:

      # Get size
      nA = int(f.readline())

      # --- Ga -------------------------------------------------------------

      Aa = np.zeros((nA, nA), dtype=int)

      i = 0
      while i<nA:

        line = f.readline().strip()

        # Skip empty lines
        if line=='': continue

        # Store data
        for j,v in enumerate(line.split()):
          Aa[i,j] = v

        # Update counter
        i += 1

      # --- Gb -------------------------------------------------------------

      Ab = np.zeros((nA, nA), dtype=int)

      i = 0
      while i<nA:

        line = f.readline().strip()

        # Skip empty lines
        if line=='': continue

        # Store data
        for j,v in enumerate(line.split()):
          Ab[i,j] = v

        # Update counter
        i += 1

      # --- Check symmetry

      # !!! TO DO !!
      directed = True

      # --- Create graphs

      Ga = Graph(nA, directed=directed, Adj=Aa)
      Gb = Graph(nA, directed=directed, Adj=Ab)

    # --- Parse solution ---------------------------------------------------

    f_sol = self.dsol + self.l_inst[n] + '.sln'
    
    with open(f_sol, 'r') as f:

      # Skip first line
      f.readline()

      S = []

      for line in f:

        if line=='': continue

        S += [int(s)-1 for s in line.strip().split()]
      
      gt = GroundTruth(Ga, Gb)
      gt.Ib = np.array(S)

    return (Ga, Gb, gt)
