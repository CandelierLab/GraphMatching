import numpy as np

class Matching:
  '''
  Matching class
  '''
    
  # === CONSTRUCTOR ========================================================

  def __init__(self, nA, nB):

    self.nA = nA
    self.nB = nB
    self.J = np.full(self.nA, None)

  def from_corr_list(self, L):
    '''
    Define the matching from a list of correspondences.
    Exemple of input: 
      [[0, 0], [1, 1], [2, 2]]
    or 
      [(0, 0), (1, 1), (2, 2)]
    '''
    
    C = np.array(L)    
    self.J[C[:,0]] = C[:,1]
