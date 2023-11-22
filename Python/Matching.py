import numpy as np

class Matching:
  '''
  Matching objects
  '''
    
  # === CONSTRUCTOR ========================================================

  def __init__(self, NetA, NetB):

    # Definitions
    self.NetA = NetA
    self.NetB = NetB
    self.nA = NetA.nNd
    self.nB = NetB.nNd

    # Matching
    self.J = np.full((self.nA), fill_value=None)

    # Measures
    self.structural_correspondence = None

  def __str__(self):
    '''
    Print function
    '''

    s = np.array2string(self.J)
    s += f'\t| SC: {self.structural_correspondence}'

    return s

  def from_corr_list(self, L):
    '''
    Define the matchinbg based on a correspondence list.
    
    Examples of correspondence lists:
      [[0,0], [1,1], [2,2]]
      [(0,0), (1,1), (2,2)]
    '''

    for c in L:
      self.J[c[0]] = c[1]

    # Compute structural correspondence
    self.get_structural_correspondence()

  def get_structural_correspondence(self):
    '''
    Compute structural correspondence
    '''

    # Matching matrix
    Z = np.full((self.nA, self.nB), False)
    for i,j in enumerate(self.J):
      if j is not None:
        Z[i,j] = True

    # Compute structural correspondence
    self.structural_correspondence = np.count_nonzero(Z @ self.NetB.Adj == self.NetA.Adj @ Z)/self.nA/self.nB

        