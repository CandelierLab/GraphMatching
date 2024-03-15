import copy
import numpy as np
from scipy import sparse
import networkx as nx

import paprint as pa

class Network:
  ''' Generic class for networks '''
    
  # === CONSTRUCTOR ========================================================

  def __init__(self, nNode=0, directed=True, nx=None):

    # Numbers
    self.nNd = nNode
    self.nNa = 0
    self.nEd = 0
    self.nEa = 0

    # Edges
    self.directed = directed
    self.edges = None

    # Adjacency matrix
    self.Adj = np.empty(0)

    # Attributes
    self.edge_attr = []
    self.node_attr = []

    # Connected
    self.isconnected = None

    # Diameter
    self.d = None

    # --- Networkx

    if nx is None:
      self.G = None
    else:
      self.import_from_networkx(nx)


  # ========================================================================
  #                             DISPLAY
  # ========================================================================

  # ------------------------------------------------------------------------
  #                             Display
  # ------------------------------------------------------------------------

  def display(self):
    ''' Display with matplotlib '''

    import matplotlib.pyplot as plt
    nx.draw(self.G)
    plt.show()

  # ------------------------------------------------------------------------
  #                          Console print
  # ------------------------------------------------------------------------

  # === PRINT ==============================================================

  def __repr__(self):
    ''' Basic info on the network'''

    pa.line(self.__class__.__name__)

    s = f'\nNumber of nodes: {self.nNd}\n'
    s += f'Number of edges: {self.nEd}\n'

    return s
  
  def print(self, maxrow=20, maxcol=20):
    '''Extended info on the network'''

    # Basic info
    print(self)

    # Adjacency matrix
    pa.matrix(self.Adj)

    # --- Network properties

    if self.isconnected is not None:
      print('The graph is {:s}connected'.format('' if self.isconnected else 'dis'))

    if self.d is not None:
      print('Diameter: {:d}'.format(self.d))

    # --- Node attributes

    for i in range(self.nNa):

      attr = self.node_attr[0]

      if 'name' in attr:
        print("\nNode attribute '{:s}' ({:s}measurable):".format(attr['name'], '' if attr['measurable'] else 'not '))
      else:
        print('\nNode attribute {:d}:'.format(i))

      print('', attr['values'])

    # --- Edge attributes

    for i in range(self.nEa):

      attr = self.edge_attr[0]

      if 'name' in attr:
        print("\nEdge attribute '{:s}' ({:s}measurable):".format(attr['name'], '' if attr['measurable'] else 'not '))
      else:
        print('\nEdge attribute {:d}:'.format(i))

      print('', attr['values'])

    print('')
    pa.line()
    print('')

  # ========================================================================
  #                             IMPORT
  # ========================================================================

  def import_from_networkx(self, G):

    # Networkx graph
    self.G = G

    # Number of nodes
    self.nNd = self.G.number_of_nodes()

    # Adjacency matrix
    self.Adj = np.full((self.nNd, self.nNd), False)
    
    E = np.array([e for e in self.G.edges])
    self.Adj[E[:,0], E[:,1]] = True
      
    # Update number of edges
    self.nEd = np.count_nonzero(self.Adj)

    # Preparation
    self.prepare()

  # ========================================================================
  #                             GENERATION
  # ========================================================================

  # ------------------------------------------------------------------------
  #                          Random structure
  # ------------------------------------------------------------------------

  def set_rand_edges(self, method='ERG', n_edges=None, p_edges=None, n_epn=None):
    '''
    NB: the parameter p can be either the number of edges n_edges (int), the 
    proportion of edges p_edges (float, in [0,1]) or the number of edges
    per node n_epn (float).
    '''

    A = np.random.rand(self.nNd, self.nNd)

    match method:

      case 'Erdös-Rényi' | 'ER':
        # In the ER model, the number of edges is guaranteed.

        # --- Define p as a number of edges

        if n_edges is not None:
          p = n_edges
        elif p_edges is not None:
          p = int(np.round(p_edges*self.nNd**2))
        elif n_epn is not None:
          p = int(np.round(n_epn*self.nNd))
        else:
          raise Exception("The proportion of edges has to be defined with at least one of the parameters: 'n_edges', 'p_deges', 'p_epn'.") 

        if p==self.nNd**2:
          self.Adj = np.full((self.nNd,self.nNd), True)
        else:
          self.Adj = A < np.sort(A.flatten())[p]

      case 'Erdös-Rényi-Gilbert' | 'ERG':
        # In the ERG the edges are drawn randomly so the exact number of 
        # edges is not guaranteed.

        # --- Define p as a proportion of edges
        if n_edges is not None:
          p = n_edges/self.nNd**2
        elif p_edges is not None:
          p = p_edges
        elif n_epn is not None:
          p = n_epn/self.nNd
        else:
          raise Exception("The proportion of edges has to be defined with at least one of the parameters: 'n_edges', 'p_deges', 'p_epn'.") 

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
    '''
    In case attr is fed directly, it should have the following structure:
    attr = {'measurable': bool, 'values': val}
    attr = {'measurable': bool, 'values': val, 'name': name}
    '''

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

    # Name
    if 'name' in kwargs:
      attr['name'] = kwargs['name']

    # Append attribute
    self.node_attr.append(attr)

    # Update number of node attributes
    self.nNa = len(self.node_attr)

  # ========================================================================
  #                             PREPARATION
  # ========================================================================

  def prepare(self):

    # Symmetrize adjacency matric for undirected graphs
    if not self.directed:
      self.Adj = np.logical_or(self.Adj, self.Adj.T)

    # Count edges
    if self.nEd==0:
      self.nEd = np.count_nonzero(self.Adj)

    # Edge list
    self.edges = np.zeros((self.nEd,2), dtype=np.int32)

    # --- Source-edge and terminus-edge matrices

    self.As = np.zeros((self.nNd, self.nEd))
    self.At = np.zeros((self.nNd, self.nEd))

    I = np.where(self.Adj)

    for i in range(len(I[0])):
      self.edges[i,:] = [I[0][i], I[1][i]]
      self.As[I[0][i], i] = 1
      self.At[I[1][i], i] = 1

    # Conversion to sparse
    # Strangely slows down when there are several matrix multiplications
    # self.As = sparse.csr_matrix(self.As)
    # self.At = sparse.csr_matrix(self.At)

    # --- Other measurements

    if self.nEd>0:
      
      # Networkx
      self.G = nx.from_numpy_array(self.Adj)

      # Connectivity
      self.isconnected = nx.is_connected(self.G)

      # Diameter
      if self.isconnected:
        self.d = nx.diameter(self.G)
      else:
        self.d = max([max(j.values()) for (i,j) in nx.shortest_path_length(self.G)])
      
  # ========================================================================
  #                             MODIFICATIONS
  # ========================================================================

  def shuffle(self):

    # New network object
    Met = copy.deepcopy(self)

    # Shuffling indexes
    Icor = np.arange(self.nNd)
    np.random.shuffle(Icor)

    # Adjacency matrix
    Met.Adj = Met.Adj[Icor, :][:, Icor]

    # Preparation
    Met.prepare()

    # --- Node attributes

    for i, attr in enumerate(Met.node_attr):
      Met.node_attr[i]['values'] = attr['values'][Icor]

    # --- Edge attributes

    if self.nEa:

      # NB: Preparation has to be done before this point.

      # Compute indexes
      J = [np.where(np.all(self.edges==[Icor[e[0]], Icor[e[1]]], axis=1))[0][0] for e in Met.edges]
      
      for i, attr in enumerate(Met.edge_attr):
        Met.edge_attr[i]['values'] = attr['values'][J]

    return (Met, Icor)

  # ========================================================================

  def subnet(self, idx):

    # Create subnetwork
    Sub = type(self)()

    # Check
    if not isinstance(idx, list) and idx>self.nNd:
      raise Exception(f"Subnetwork: The number of nodes in the subnet ({idx}) is greater than in the original network ({self.nNd})")

    # Indexes
    I = idx if isinstance(idx, list) else np.random.choice(range(self.nNd), idx, replace=False)
    # I = idx if isinstance(idx, list) else random.sample(range(self.nNd), idx)
    K = np.ix_(I,I)

    # Adjacency matrix
    Sub.Adj = self.Adj[K]

    # --- Properties

    Sub.nNd = len(I)
    Sub.nEd = np.count_nonzero(Sub.Adj)
    Sub.nNa = self.nNa
    Sub.nEa = self.nEa

     # Preparation
    Sub.prepare()

    # --- Node attributes

    for attr in self.node_attr:

      Sub.add_node_attr( {'measurable': attr['measurable'], 'values': attr['values'][I]} )
    
    # --- Edge attributes

    if self.nEa:

      # NB: Preparation has to be done before this point.

      # Compute indexes
      J = [np.where(np.all(self.edges==[I[e[0]], I[e[1]]], axis=1))[0][0] for e in Sub.edges]
      
      for attr in self.edge_attr:
        Sub.add_edge_attr( {'measurable': attr['measurable'], 'values': attr['values'][J]} )

    return Sub if isinstance(idx, list) else (Sub, I)
  
  # ========================================================================

  def degrade(self, type, **kwargs):
    '''Network degradation

    Degradation can be done in many different ways ('type' argument):
    - Structure:
      'Rn': Remove nodes (and the corresponding edges)
      'Re': Remove edges
      'Ces': Change edge sources
      'Cet': Change edge targets
      'Cest': Change edge sources and targets
    - Attributes:
      'Cna': Change node attributes
      'Cea': Change edge attributes
      'Nna': add Gaussian noise to node attribute
      'Nea': add Gaussian noise to edge attribute
    '''

    # !!! TO RECODE !!!

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
