'''
QAPLIB wrapper
'''

import os
import numpy as np

import project
from Graph import *

class Instance:

  def __init__(self, name, n, A, B, s, score):

    self.name = name
    self.n = n
    self.A = A
    self.B = B
    self.s = s
    self.score = score

class QAPLIB:

  def __init__(self):
    
    # Paths
    self.root = project.root + '/Files/QAPLIB/'
    self.dinst = self.root + 'Instances/'
    self.dsol = self.root + 'Solutions/'

    # List instances with solutions
    self.l_inst = sorted([s[:-4] for s in os.listdir(self.dsol)])
    self.n_inst = len(self.l_inst)

  def get(self, id):
    '''
    Get n-th instance and solution
    '''

    if isinstance(id, str):
      iname = id

    else:
      if id >= self.n_inst:
        raise Exception(f'Please choose a solution in [0,{self.n_inst-1}].') 
      
      iname = self.l_inst[id]
    
    # --- Parse instance ---------------------------------------------------

    f_inst = self.dinst + iname + '.dat'
    
    with open(f_inst, 'r') as f:

      # Get size
      n = int(f.readline())

      # --- A --------------------------------------------------------------

      A = np.zeros((n, n), dtype=int)

      i = 0
      while i<n:

        V = []
        while len(V)<n:

          line = f.readline().strip()

          # Skip empty lines
          if line=='': continue

          V += line.split()

        # Store data
        for j,v in enumerate(V):
          A[i,j] = v

        # Update counter
        i += 1

      # --- B --------------------------------------------------------------

      B = np.zeros((n, n), dtype=int)

      i = 0
      while i<n:

        V = []
        while len(V)<n:

          line = f.readline().strip()

          # Skip empty lines
          if line=='': continue

          V += line.split()

        # Store data
        for j,v in enumerate(V):
          B[i,j] = v

        # Update counter
        i += 1

    # --- Parse solution ---------------------------------------------------

    f_sol = self.dsol + iname + '.sln'
    
    with open(f_sol, 'r') as f:

      # First line
      score = int(f.readline().strip().split()[1])

      s = []

      for line in f:

        if line=='': continue

        s += [int(x)-1 for x in line.strip().split()]
    
    return Instance(iname, n, A, B, s, score)

  def get_graphs(self, id, precision=(None, None)):
    '''
    Get n-th instance and solution
    '''

    I = self.get(id)

    # --- Check symmetry

    directed = not ((I.A==I.A.T).all() and (I.B==I.B.T).all())

    # --- Create graphs

    Ga = Graph(I.A.shape[0], directed=directed, Adj=I.A>0)
    Gb = Graph(I.B.shape[0], directed=directed, Adj=I.B>0)

    # --- Edge attributes

    Ga.add_edge_attr({'measurable': True,
                      'error':precision[0],
                      'values':[I.A[e[0], e[1]] for e in Ga.edges]})
    
    Gb.add_edge_attr({'measurable': True,
                      'error':precision[1],
                      'values':[I.B[e[0], e[1]] for e in Gb.edges]})

    # --- Ground truth

    gt = GroundTruth(Ga, Gb)
    gt.Ib = np.array(I.s)

    return (Ga, Gb, gt)
