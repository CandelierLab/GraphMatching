import random
import copy
import numpy as np

class Network:
  ''' Generic class for networks '''
    
  # === CONSTRUCTOR ========================================================

  def __init__(self, nNode=0):

    # Numbers
    self.nNd = nNode
    self.nNa = 0
    self.nEd = 0
    self.nEa = 0

    # Edge list
    self.edges = None

    # Adjacency matrix
    self.Adj = np.empty(0)

    # Attributes
    self.edge_attr = []
    self.node_attr = []

  # === PRINT ==============================================================

  def __repr__(self):
    ''' Basic info on the network'''

    s = '-'*50 + '\n'
    s += self.__class__.__name__ + ' network\n\n'

    s += f'Number of nodes: {self.nNd}\n'
    s += f'Number of edges: {self.nEd}\n'

    return s
  
  def print(self, maxrow=20, maxcol=20):
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

    # --- Edge attributes

    for i in range(self.nEa):

      attr = self.edge_attr[0]

      if 'name' in attr:
        print("\nEdge attribute '{:s}' ({:s}measurable):".format(attr['name'], '' if attr['measurable'] else 'not '))
      else:
        print('\nEdge attribute {:d}:'.format(i))

      print('', attr['values'])

    # --- Node attributes

    for i in range(self.nNa):

      attr = self.node_attr[0]

      if 'name' in attr:
        print("\nNode attribute '{:s}' ({:s}measurable):".format(attr['name'], '' if attr['measurable'] else 'not '))
      else:
        print('\nNode attribute {:d}:'.format(i))

      print('', attr['values'])

    print('')

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

    # Prepare structure
    self.prepare()
    
  # ------------------------------------------------------------------------
  #                              Attributes
  # ------------------------------------------------------------------------

  def add_edge_attr(self, *args, **kwargs):

    if isinstance(args[0], str):

      match args[0]:

        case 'rand':

          # Parameters
          mv = kwargs['min'] if 'min' in kwargs else 0
          Mv = kwargs['max'] if 'max' in kwargs else 1

          # Attribute
          attr = {'measurable': True, 
                  'values': np.random.random(self.nEd)*(Mv-mv) + mv}

        case 'gauss':
          
          # Parameters
          mu = kwargs['mean'] if 'mean' in kwargs else 0
          sigma = kwargs['std'] if 'std' in kwargs else 1

          # Attribute
          attr = {'measurable': True, 
                  'values': mu + sigma*np.random.randn(self.nEd)}

    else:
      
      attr = args[0]

    if 'name' in kwargs:
      attr['name'] = kwargs['name']

    # Append attribute
    self.edge_attr.append(attr)

    # Update number of edge attributes
    self.nEa = len(self.edge_attr)

  # ------------------------------------------------------------------------

  def add_node_attr(self, *args, **kwargs):

    if isinstance(args[0], str):

      match args[0]:

        case 'rand':

          # Parameters
          mv = kwargs['min'] if 'min' in kwargs else 0
          Mv = kwargs['max'] if 'max' in kwargs else 1

          # Attribute
          attr = {'measurable': True, 
                  'values': np.random.random(self.nNd)*(Mv-mv) + mv}

        case 'gauss':
          
          # Parameters
          mu = kwargs['mean'] if 'mean' in kwargs else 0
          sigma = kwargs['std'] if 'std' in kwargs else 1

          # Attribute
          attr = {'measurable': True, 
                  'values': mu + sigma*np.random.randn(self.nNd)}

    else:
      
      attr = args[0]

    if 'name' in kwargs:
      attr['name'] = kwargs['name']

    # Append attribute
    self.node_attr.append(attr)

    # Update number of node attributes
    self.nNa = len(self.node_attr)

  # ========================================================================
  #                             PREPARATION
  # ========================================================================

  def prepare(self, force=False):

    # Edge list
    self.edges = np.zeros((self.nEd,2), dtype=np.int32)

    # Source-edge and terminus-edge matrices
    self.As = np.zeros((self.nNd, self.nEd))
    self.At = np.zeros((self.nNd, self.nEd))

    I = np.where(self.Adj)

    for i in range(len(I[0])):
      self.edges[i,:] = [I[0][i], I[1][i]]
      self.As[I[0][i], i] = 1
      self.At[I[1][i], i] = 1
    
  # ========================================================================
  #                             MODIFICATIONS
  # ========================================================================

  def shuffle(self):

    # New network object
    Met = copy.deepcopy(self)

    # Suffling indexes
    Icor = np.arange(self.nNd)
    np.random.shuffle(Icor)

    # Adjacency matrix
    Met.Adj = Met.Adj[Icor, :][:, Icor]

    # Attributes
    # --- TO DO ---

    # Preparation
    # (Source-edge and terminus-edge matrices)
    Met.prepare()

    return (Met, Icor)

  # ========================================================================

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
    Sub.nNa = self.nNa
    Sub.nEa = self.nEa

    # Attributes
    for i in range(self.nEa):
      Sub.add_edge_attr(self.edge_attr[i][I])

    for i in range(self.nNa):
      Sub.add_node_attr(self.node_attr[i][I])
    
    # Preparation
    Sub.prepare()

    return Sub if isinstance(idx, list) else (Sub, I)
  
  # ========================================================================

  def degrade(self, type, **kwargs):
    '''Network degradation
    Degradation can be done in several different ways:
    - Structure: edges are reassigned, i.e. ones in the adjacency matrix are
        moved to a different place. Corresponding attributes may be 
        reassigned to a new value as well (type='struct+attr') or not (type='struct').
    - Attributes: Values are reassigned (type='attr').

    Parameters:
      type='struct'
        p: proportion of edges to reassign (default is undefined)
        n: number of edges to reassign (default n=1)

      type='struct+attr'
        TO DO

      type='attr'
        TO DO
    '''

    # New network object
    Det = copy.deepcopy(self)

    match type:

      case 'struct':
        
        # Number of modifications
        if 'p' in kwargs:
          n = round(kwargs['p']*Det.nEd)
        else:
          n = kwargs['n'] if 'n' in kwargs else 1

        # --- Edges to remove

        I = np.where(Det.Adj)
        nEtr = len(I[0])

        if n>nEtr:
          raise Exception('Not enough edges to modify ({:d} asked for {:d} existing).'.format(n, nEtr)) 
        
        # --- Edges to create

        J = np.where(Det.Adj==False)
        nEtc = len(J[0])

        if n>nEtc:
          raise Exception('Too many edges to modify ({:d} asked for {:d} possible).'.format(n, nEtc)) 

        # --- Operation

        # Remove edges
        K = np.random.choice(nEtr, n, replace=False)
        Det.Adj[I[0][K], I[1][K]] = False

        # Unmodified edges
        Jcor = np.setdiff1d(range(Det.nEd), K)

        # New edges
        K = np.random.choice(nEtc, n, replace=False)
        Det.Adj[J[0][K], J[1][K]] = True

      case 'struct+attr':
        pass

      case 'attr':
        pass

    # Preparation
    Det.prepare()

    return (Det, Jcor)
