import random
import numpy as np

# === Generic network ======================================================

class Generic:
  ''' Generic class for networks '''
    
  def __init__(self):

    self.nNd = 0
    self.nEd = 0

    # Adjacency matrices
    self.Adj = np.empty(0) 
    self.Bdj = np.empty(0)

  def __repr__(self):
    ''' String representation of the network '''

    s = '-'*50 + '\n'
    s += self.__class__.__name__ + ' network\n\n'

    s += f'Number of nodes: {self.nNd}\n'
    s += f'Number of edges: {self.nEd}\n'

    return s
  
  def subnet(self, nNd, method='random'):

    # Create subnetwork
    Sub = type(self)()

    # Check
    if nNd>self.nNd:
      raise Exception(f"Subnetworking: The number of nodes in the subnet ({nNd}) is greater than in the original network ({self.nNd})")

    match method:

      case 'random':

        # Draw random set of Nodes
        I = random.sample(range(self.nNd), nNd)
        K = np.ix_(I,I)

        # Adjacency matrices
        Sub.Adj = self.Adj[K]
        Sub.Bdj = self.Bdj[K]

        # Numbers
        Sub.nNd = nNd
        Sub.nEd = np.count_nonzero(Sub.Bdj)

    return Sub

# === Random Network =======================================================

class Random(Generic):
  ''' Erdos-Renyi network '''

  def __init__(self, N=None, p=None, method='Erdös-Rényi'):
    
    super().__init__()

    # Empty network
    if p is None: return

    # Set number of nodes
    self.nNd = N

    # --- Binary adjacency matrix

    A = np.random.rand(self.nNd, self.nNd)

    match method:
      case 'Erdös-Rényi' | 'ER':
        # In the ER model, the number of edges is guaranteed.
        # NB: the parameter p can be either the number of edges (int)
        #   or a proportion of edges (float, in [0,1])

        # In case p is a proportion, convert it to an integer
        if isinstance(p, float):
          p = int(np.round(p*self.nNd**2))

        # Define edges
        self.Bdj = (A < np.sort(A.flatten())[p])

      case 'Erdös-Rényi-Gilbert' | 'ERG':
        # In the ERG the edges are drawn randomly so the exact number of 
        # edges is not guaranteed.

        self.Bdj = None

    # Update number of edges
    self.nEd = np.count_nonzero(self.Bdj)

    # --- Weighted adjacency matrix

    self.Adj = self.Bdj.astype(float)

