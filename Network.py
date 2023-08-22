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
    ''' String representation of the network '''

    s = '-'*50 + '\n'
    s += self.__class__.__name__ + ' network\n\n'

    s += f'Number of nodes: {self.nNd}\n'
    s += f'Number of edges: {self.nEd}\n'

    return s
  
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
    self.nEd = len(self.edge)
    
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
  
# === Random Network =======================================================

class Random(Network):
  ''' Erdos-Renyi network '''

  def __init__(self, N=None, p=None, method='rand'):
    
    super().__init__()

    # Empty network
    if p is None: return

    # Set number of nodes
    self.nNd = N

    # --- Binary adjacency matrix

    A = np.random.rand(self.nNd, self.nNd)

    match method:
      case 'rand':
        # In this methods the weights are drawn at random uniformly over 
        # the range [0,1]

        self.Adj = A

      case 'Erdös-Rényi' | 'ER':
        # In the ER model, the number of edges is guaranteed.
        # NB: the parameter p can be either the number of edges (int)
        #   or a proportion of edges (float, in [0,1])

        # In case p is a proportion, convert it to an integer
        if isinstance(p, float):
          p = int(np.round(p*self.nNd**2))

        # Define edges
        Bdj = (A < np.sort(A.flatten())[p])

        # Weighted adjacency matrix
        self.Adj = Bdj.astype(float)

      case 'Erdös-Rényi-Gilbert' | 'ERG':
        # In the ERG the edges are drawn randomly so the exact number of 
        # edges is not guaranteed.

        self.Adj = None

    # Prepare
    self.prepare()

