import os
import numpy as np

# === MATCHING =============================================================

class Matching:
  '''
  Matching object
  '''

  # ========================================================================
  # |                                                                      |
  # |                          Constructor                                 |
  # |                                                                      |
  # ========================================================================

  def __init__(self, NetA, NetB):

    # Definitions
    self.NetA = NetA
    self.NetB = NetB
    self.nA = NetA.nNd
    self.nB = NetB.nNd

    # Matching
    self.idxA = np.empty(0)
    self.idxB = np.empty(0)

    # Measures
    self.score = None
    self.accuracy = None
    self.structural_quality = None

  # ========================================================================
  # |                                                                      |
  # |                             Display                                  |
  # |                                                                      |
  # ========================================================================

  def __str__(self):
    '''
    Print function
    '''

    # --- Parameters -------------------------------------------------------

    # max caracters per line 
    mcpl = os.get_terminal_size()[0]

    # Maximum number of correspondences to display
    kmax = 100

    s = '╒══ Matching ' + '═'*(mcpl-13) + '\n'
    
    s += '│\n'
    param_suff = ''
  
    # Matching score
    if self.score is not None:
      s += f'│ Matching score: {self.score:.03f}\n'
      param_suff = '│\n'

    # Accuracy
    if self.accuracy is not None:
      s += f'│ Accuracy: {self.accuracy:.03f}\n'
      param_suff = '│\n'

    # Structural quality
    if self.structural_quality is not None:
      s += f'│ Structural quality: {self.structural_quality:.03f}\n'
      param_suff = '│\n'

    s += param_suff
 
    # --- Correspondences --------------------------------------------------

    # --- Number of correspondences to display

    km = self.idxA.size

    if not km:

      s += '│ Empty list of correspondences.\n'
      
    else:

      if km > kmax:
        km = kmax
        suff = f'│ ... and {self.idxA.size-kmax} more.\n'
      else:
        suff = ''

      # --- Display correspondences
      
      l0 = '│ '
      l1 = '│ '
      l2 = '│ '

      for k in range(km):

        # --- Buffers

        bl = len(f'{max(self.idxA[k], self.idxB[k])}')

        b0 = f' \033[7m {self.idxA[k]:{bl}d} \033[0m'
        b1 = f'  {self.idxB[k]:{bl}d} '
        b2 = '─'*(bl+3)

        if len(l1+b1)<mcpl:

          l0 += b0
          l1 += b1
          l2 += b2

        else:

          # Flush
          s += l0 + '\n'
          s += l1 + '\n'
          s += l2 + '─\n'

          # Restart
          l0 = '│ ' + b0 
          l1 = '│ ' + b1
          l2 = '│ ' + b2

      # Flush
      s += l0 + '\n'
      s += l1 + '\n'

      s += suff

    s += '╘' + '═'*(mcpl-1) + '\n'

    return s

  # ========================================================================
  # |                                                                      |
  # |                             Imports                                  |
  # |                                                                      |
  # ========================================================================

  def from_lists(self, idxA, idxB, sort=True, initialization=True):
    '''
    Define the matching based on two lists of indices.
    '''

    self.idxA = np.array(idxA) if isinstance(idxA, list) else idxA
    self.idxB = np.array(idxB) if isinstance(idxB, list) else idxB

    # Sort
    if sort:
      I = self.idxA.argsort()
      self.idxA = self.idxA[I]
      self.idxB = self.idxB[I]

    # Initialization
    if initialization:
      self.initialize()

  # ========================================================================
  # |                                                                      |
  # |                          Computations                                |
  # |                                                                      |
  # ========================================================================

  def initialize(self):
    '''
    Initialization
    '''

    # Structural quality
    if self.structural_quality is None:
      self.compute_structural_quality()


  def compute_structural_quality(self):
    '''
    Compute the structural quality of the matching
    '''

    # Matching matrix
    Z = np.full((self.nA, self.nB), False)
    for (i,j) in zip(self.idxA, self.idxB):
      if j is not None:
        Z[i,j] = True

    # Compute structural correspondence
    self.structural_quality = np.count_nonzero(Z @ self.NetB.Adj == self.NetA.Adj @ Z)/self.nA/self.nB

  def compute_accuracy(self, Idx):
    '''
    Compute the matching accuracy
    '''

    self.accuracy= np.count_nonzero(Idx[self.idxB]==self.idxA)/self.idxA.size

  def compute_score(self, X):
    '''
    Compute the matching score
    '''

    self.score = np.sum(X[self.idxA, self.idxB])

# === MATCHING SET =========================================================

class MatchingSet:
  '''
  Set of matchings.
  '''

  def __init__(self):
    pass