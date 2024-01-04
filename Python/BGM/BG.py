
import numpy as np
from scipy.sparse import coo_array

import paprint as pa

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

class Phi:

  def __init__(self, keys=None, values=None):

    self.k = np.array([], dtype=int) if keys is None else keys
    self.v = np.ones(self.k.shape[0], dtype=int) if values is None else values

  def __str__(self):

    s = f'\n--- Keys:\n{self.k}\n\n---Values:\n{self.v}'

    return s
  
  def nu(self):
    return self.v.size
  
  def mu(self):
    return np.sum(self.v)

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

    phi = Phi(np.array([np.insert(np.setdiff1d(range(self.beta), i), 0, i) for i in range(self.beta)]))

    # --- Iterations

    nOp = 0

    # Display
    disp = lambda x : print(f'nu={x.nu()} ; mu={x.mu()}')
    # disp(phi)

    for eta in range(1, self.alpha):

      # pa.line(f'eta = {eta}')

      # --- Relaxation

      if eta==self.alpha-1:

        # On the last step, there is nothing left to relax
        phi_ = phi

      else:

        key = [phi.k]

        for u in range(1, self.beta-eta):
          key.append(np.hstack((phi.k[:,0:u], phi.k[:,u+1:], phi.k[:,[u]])))
          if u==1:
            val = np.concatenate((phi.v, phi.v))
          else:
            val = np.concatenate((val, phi.v))


        phi_ = Phi(np.vstack(key), val)

      # --- Compression

      nOp += phi_.nu()

      z = {}

      for i, k in enumerate(phi_.k):

        if self.A[eta-1,k[0]] and self.A[eta,k[-1]]:

          t = tuple(np.concatenate(([k[-1]], k[1:-1])))

          if t in z:
            z[t] += phi_.v[i]
          else:
            z[t] = phi_.v[i]

      phi = Phi(keys=np.array(list(z.keys())), values=np.array(list(z.values()), dtype=int))

      # disp(phi)
      
    return np.sum(phi.v), nOp


