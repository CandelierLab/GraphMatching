
import numpy as np
from scipy.sparse import coo_array

# ==========================================================================

class BinaryGraph:
  '''
  Binary Graph
  '''

  def __init__(self, a, q,  b=None):

    # Definitions
    self.a = a
    self.b = a if b is None else b
    self.q = q

    self.alpha = min(self.a, self.b)
    self.beta = max(self.a, self.b)

    # Checks
    assert q>=2*self.alpha, 'q is too low'
    assert q<=self.a*self.b, 'q is too high'

    # Define Block matrix
    

# ==========================================================================

class Block:
  '''
  Block matrix
  '''

  def __init__(self, A):

    if A.shape[0]<=A.shape[1]:
      self.A = A
    else:
      self.A = A.T

    self.alpha, self.beta = self.A.shape

  def brute_force(self):
    '''
    Brute force solution
    '''

    # Recursive function
    def get_subs(M, base=[], idx=None):

      if idx is None:
        idx = range(M.shape[0])

      if M.size==1:
        return [idx] if M[0] else []

      else:
        R = []

        if np.count_nonzero(M[:,0]):

          for j in np.where(M[:,0])[0]:

            M_ = M[np.setdiff1d(np.arange(M.shape[1]),j), 1:]
            idx_ = [idx[i] for i in np.setdiff1d(list(range(len(idx))), j)]

            for s in get_subs(M_, base + [idx[j]], idx_):
              R.append([idx[j]] + s)

        return R

    # List all solutions
    S = get_subs(self.A)

    return S, len(S)
    
  def Uno(self):


    pass

  def RC(self):

    # --- Define phi_1

    phi = np.array([np.insert(np.setdiff1d(range(self.beta), i), 0, i) for i in range(self.beta)])

    # --- Iterations

    print(phi)

    for eta in range(1):

      # --- Relaxation

      tmp = [phi]
      for k in range(1, self.beta-eta-1):
        tmp.append(np.hstack((phi[:,0:k], phi[:,k+1:], phi[:,[k]])))

      phi_ = np.vstack(tmp)

      # --- Compression

      # Transition matrix
      M = 

      print(len(phi))



