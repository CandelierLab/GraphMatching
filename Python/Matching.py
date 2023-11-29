import numpy as np

# === MATCHING =============================================================

class Matching:
  '''
  Matching objects
  '''

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

# === MATCHING SET =========================================================

class MatchingSet:
  '''
  Set of matchings.
  '''

  def __init__(self, NetA, NetB, M, all_matchings=False):

    # Definitions
    self.NetA = NetA
    self.NetB = NetB

    # Total undetermination case
    self.all_matchings = all_matchings

    # Matching list
    self.matchings = []
    if not self.all_matchings:      
      for m in M:
        tmp = Matching(self.NetA, self.NetB)
        tmp.from_corr_list(m)
        self.matchings.append(tmp)

    # Accuracy
    self.accuracy = None

  def __str__(self):
    
    if self.all_matchings:

      s = 'All possible matching (total undetermination).\n'

    else:

      s = f'{len(self.matchings)} Matchings:\n'

      for m in self.matchings:
        s += m.__str__() + '\n'

    # --- Accuracy
    if self.accuracy is not None:
      s += f'\nSet accuracy: {self.accuracy}\n'

    return s
  
  def compute_accuracy(self, Icor):
    '''
    Compute the matching set accuracy based on the set correspondence
    '''

    if self.all_matchings:

      self.accuracy = 1/min(self.NetA.nNd, self.NetB.nNd)

    else:

      count = 0
      total = 0
      for m in self.matchings:
        for (i,j) in enumerate(m.J):
          if j is not None:
            total += 1
            if Icor[j]==i:
              count += 1

      self.accuracy = count/total