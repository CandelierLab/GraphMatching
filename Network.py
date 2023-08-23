import random
import numpy as np

class Network:
  ''' Generic class for networks '''
    
  # === CONSTRUCTOR ========================================================

  def __init__(self, nNode=0):

    self.nNd = nNode
    self.nEd = 0

    # Lists of nodes and edges
    self.node = None
    self.edge = None

    # Adjacency matrix
    self.Adj = np.empty(0)

  # === PRINT ==============================================================

  def __repr__(self):
    ''' Basic info on the network'''

    s = '-'*50 + '\n'
    s += self.__class__.__name__ + ' network\n\n'

    s += f'Number of nodes: {self.nNd}\n'
    s += f'Number of edges: {self.nEd}\n'

    return s
  
  def print(self, maxrow=10, maxcol=10):
    '''Extended info on the network'''

    # Basic info
    print(self)

    # --- Header

    r = ['    ', '    ', '    ']
    for j in range(self.Adj.shape[1]):

      if j>maxcol: 
        r[-1] += ' ...'
        break

      d, u = divmod(j, 10)
      r[0] += ' {:d}'.format(d)
      r[1] += ' {:d}'.format(u)
      r[-1] += '--'

    print(r[0])
    print(r[1])
    print(r[-1])

    # --- Rows

    for i, row in enumerate(self.Adj):

      if i>maxrow: 
        print('...')
        break

      print('{:02d} |'.format(i), end='')
      for j, a in enumerate(row):
        if j>maxcol: 
          print('    ', end='')
          break
        print(' 1' if a else ' .', end='')
      print(' |')

    print(r[-1])


  # ========================================================================
  #                             GENERATION
  # ========================================================================

  # ------------------------------------------------------------------------
  #                          Random structure
  # ------------------------------------------------------------------------

  def set_rand_edges(self, method='ERG', p=0.5):
    
    A = np.random.rand(self.nNd, self.nNd)

    match method:

      case 'Erdös-Rényi' | 'ER':
        # In the ER model, the number of edges is guaranteed.
        # NB: the parameter p can be either the number of edges (int)
        #   or a proportion of edges (float, in [0,1])

        # In case p is a proportion, convert it to an integer
        if isinstance(p, float):
          p = int(np.round(p*self.nNd**2))

        self.Adj = A < np.sort(A.flatten())[p]

      case 'Erdös-Rényi-Gilbert' | 'ERG':
        # In the ERG the edges are drawn randomly so the exact number of 
        # edges is not guaranteed.

        self.Adj = A < p

    # Update number of edges
    self.nEd = np.count_nonzero(self.Adj)
    
  # ========================================================================
  #                             PREPARATION
  # ========================================================================

  def prepare(self, force=False):

    # --- Edges

    if self.edge is None or force:

      self.edge = []

      for ij in np.argwhere(self.Adj):
        self.edge.append({'i': ij[0], 'j': ij[1], 'w': float(self.Adj[ij[0], ij[1]])})

    # --- Source-edge and terminus-edge matrices, weight vectors

    # Net A
    self.As = np.zeros((self.nNd, self.nEd))
    self.At = np.zeros((self.nNd, self.nEd))
    for k, e in enumerate(self.edge):
      self.As[e['i'], k] = 1
      self.At[e['j'], k] = 1
    
  def subnet(self, idx):

    # Create subnetwork
    Sub = type(self)()

    # Check
    if not isinstance(idx, list) and idx>self.nNd:
      raise Exception(f"Subnetworking: The number of nodes in the subnet ({idx}) is greater than in the original network ({self.nNd})")

    # Indexes
    I = idx if isinstance(idx, list) else random.sample(range(self.nNd), idx)
    K = np.ix_(I,I)

    # Adjacency matrices
    Sub.Adj = self.Adj[K]

    # Numbers
    Sub.nNd = len(I)
    Sub.nEd = np.count_nonzero(Sub.Adj)
    
    return Sub if isinstance(idx, list) else (Sub, I)
  