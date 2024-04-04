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

  def __init__(self, Ga, Gb, algorithm=None):

    # Definitions
    self.Ga = Ga
    self.Gb = Gb
    self.nA = Ga.nV
    self.nB = Gb.nV

    # Matching
    self.idxA = np.empty(0)
    self.idxB = np.empty(0)

    # Measures
    self.score = None
    self.accuracy = None
    self.structural_quality = None

    # Misc infos
    self.algorithm = algorithm
    self.time = None
    self.info = None

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
    kmax = 20

    s = '╒══ Matching ' + '═'*(mcpl-13) + '\n'
    
    s += '│\n'
    param_suff = ''
  
    # Algorithm
    if self.algorithm is not None:
      s += f'│ Algorithm: \t\t\033[31m{self.algorithm}\033[0m\n'
      param_suff = '│\n'

    # Computation time
    if self.time is not None:

      s += f'│ Computation time: \t\033[35m{self.time["total"]:0.2f}\033[0m ms'

      if self.algorithm in ['Zager', 'GASM']:
        s += f' ({self.time["scores"]:0.2f} + {self.time["LAP"]:0.2f})'

      s += '\n'
      param_suff = '│\n'

    # Matching score
    if self.score is not None:
      s += f'│ Matching score: \t{self.score:.03f}\n'
      param_suff = '│\n'

    # Accuracy
    if self.accuracy is not None:
      s += f'│ Accuracy: \t\t\033[36m{self.accuracy:.03f}\033[0m\n'
      param_suff = '│\n'

    # Structural quality
    if self.structural_quality is not None:
      s += f'│ Structural quality: \t\033[96m{self.structural_quality:.03f}\033[0m\n'
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

    if not self.nA or not self.nB:

      self.structural_quality = 0

    else:

      # Matching matrix
      Z = np.full((self.nA, self.nB), False)
      for (i,j) in zip(self.idxA, self.idxB):
        if j is not None:
          Z[i,j] = True

      # Compute structural correspondence
      self.structural_quality = np.count_nonzero(Z @ self.Gb.Adj == self.Ga.Adj @ Z)/self.nA/self.nB

  def compute_accuracy(self, gt=None):
    '''
    Compute the matching accuracy
    '''
    
    if not self.nA or not self.nB:
      self.accuracy = 0

    elif gt is None:
      self.accuracy = np.count_nonzero(self.idxA==self.idxB)/self.nA

    else:

      # self.accuracy = np.count_nonzero(gt.Idx[self.idxB]==self.idxA)/self.nA

      if self.nA >= self.nB:
        self.accuracy = np.count_nonzero(self.idxA[gt.Ib]==gt.Ia[self.idxB])/self.nB
      else:
        self.accuracy = np.count_nonzero(self.idxB[gt.Ia]==gt.Ib[self.idxA])/self.nA

  def compute_score(self, X):
    '''
    Compute the matching score
    '''

    self.score = np.sum(X[self.idxA, self.idxB])
