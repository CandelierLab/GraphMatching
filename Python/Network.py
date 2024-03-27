import copy
import numpy as np
from scipy import sparse
import networkx as nx

import paprint as pa

# ##########################################################################
#                          Generic Network class
# ##########################################################################

class Network:
  ''' Generic class for networks '''
    
  # === CONSTRUCTOR ========================================================

  def __init__(self, nV=0, directed=True, nx=None):

    # Numbers
    self.nV = nV
    self.nVa = 0
    self.nE = 0
    self.nEa = 0

    # Edges
    self.directed = directed
    self.edges = None

    # Adjacency matrix
    self.Adj = np.empty(0)

    # Attributes
    self.edge_attr = []
    self.vrtx_attr = []

    # Connected
    self.is_strongly_connected = None

    # Diameter
    self.d = None

    # --- Networkx

    if nx is None:
      self.nx = None
      self.nxu = None
    else:
      self.import_from_networkx(nx)


  # ========================================================================
  #                             DISPLAY
  # ========================================================================

  # ------------------------------------------------------------------------
  #                             Display
  # ------------------------------------------------------------------------

  def display(self):
    '''
    Display with matplotlib
    '''

    import matplotlib.pyplot as plt

    nx.draw(self.nx)
    
    plt.show()

  # ------------------------------------------------------------------------
  #                          Console print
  # ------------------------------------------------------------------------

  def __repr__(self):
    ''' 
    Some info on the network
    '''

    pa.line(self.__class__.__name__)

    s = f'\nNumber of vertices: {self.nV}\n'
    s += f'Number of edges: {self.nE}\n'

    return s
  
  def print(self, maxrow=20, maxcol=20):
    '''Extended info on the network'''

    # Basic info
    print(self)

    # Adjacency matrix
    pa.matrix(self.Adj)

    # --- Network properties

    if self.is_strongly_connected is not None:
      print('The graph is strongly {:s}connected'.format('' if self.is_strongly_connected else 'dis'))

    if self.d is not None:
      print('Diameter: {:d}'.format(self.d))

    # --- Vertex attributes

    for i in range(self.nVa):

      attr = self.vrtx_attr[0]

      if 'name' in attr:
        print("\nVertex attribute '{:s}' ({:s}measurable):".format(attr['name'], '' if attr['measurable'] else 'not '))
      else:
        print('\nVertex attribute {:d}:'.format(i))

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

    # Assign netorkx graph
    self.nx = G

    # Directivity
    self.directed = G.is_directed()

    # Undirected flavor
    self.nxu = G.to_undirected() if self.directed else self.nx

    # Adjacency matrix
    self.Adj = nx.to_numpy_array(self.nx, dtype=bool)

    # Number of vertices
    self.nV = self.nx.number_of_nodes()

    # Number of edges
    self.nE = self.nx.number_of_edges()

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

    A = np.random.rand(self.nV, self.nV)

    match method:

      case 'Erdös-Rényi' | 'ER':
        # In the ER model, the number of edges is guaranteed.

        # --- Define p as a number of edges

        if n_edges is not None:
          p = n_edges
        elif p_edges is not None:
          p = int(np.round(p_edges*self.nV**2))
        elif n_epn is not None:
          p = int(np.round(n_epn*self.nV))
        else:
          raise Exception("The proportion of edges has to be defined with at least one of the parameters: 'n_edges', 'p_deges', 'p_epn'.") 

        if p==self.nV**2:
          self.Adj = np.full((self.nV,self.nV), True)
        else:
          self.Adj = A < np.sort(A.flatten())[p]

      case 'Erdös-Rényi-Gilbert' | 'ERG':
        # In the ERG the edges are drawn randomly so the exact number of 
        # edges is not guaranteed.

        # --- Define p as a proportion of edges
        if n_edges is not None:
          p = n_edges/self.nV**2
        elif p_edges is not None:
          p = p_edges
        elif n_epn is not None:
          p = n_epn/self.nV
        else:
          raise Exception("The proportion of edges has to be defined with at least one of the parameters: 'n_edges', 'p_deges', 'p_epn'.") 

        self.Adj = A < p

    # Update number of edges
    self.nE = np.count_nonzero(self.Adj)

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
                  'values': np.random.random(self.nE)*(Mv-mv) + mv}

        case 'gauss':
          
          # Parameters
          mu = kwargs['mean'] if 'mean' in kwargs else 0
          sigma = kwargs['std'] if 'std' in kwargs else 1

          # Attribute
          attr = {'measurable': True, 
                  'values': mu + sigma*np.random.randn(self.nE)}

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
                  'values': np.random.random(self.nV)*(Mv-mv) + mv}

        case 'gauss':
          
          # Parameters
          mu = kwargs['mean'] if 'mean' in kwargs else 0
          sigma = kwargs['std'] if 'std' in kwargs else 1

          # Attribute
          attr = {'measurable': True, 
                  'values': mu + sigma*np.random.randn(self.nV)}

    else:
      
      attr = args[0]

    # Name
    if 'name' in kwargs:
      attr['name'] = kwargs['name']

    # Append attribute
    self.vrtx_attr.append(attr)

    # Update number of node attributes
    self.nVa = len(self.vrtx_attr)

  # ========================================================================
  #                             PREPARATION
  # ========================================================================

  def prepare(self):
    '''
    Prepares the network for comparison by:
    - Compute the graph connected state
    - Computing the graph diameter
    - Computing the As and At matrices.

    Also: 
    - establish the list of edges if it is empty
    - Perform other measurements:
      + strong connectivity (is every vertex reachable from any vertex)
      + diameter
    '''

    # --- Preparation

    # Edge list
    list_edges = self.edges is None

    if list_edges:
      self.edges = np.zeros((self.nE, 2), dtype=np.int32)

    # Source-edge and terminus-edge matrices
    self.As = np.zeros((self.nV, self.nE))
    self.At = np.zeros((self.nV, self.nE))

    # --- Loop through edges

    I = np.where(self.Adj)

    for i in range(len(I[0])):

      self.As[I[0][i], i] = 1
      self.At[I[1][i], i] = 1

      if list_edges:
        self.edges[i,:] = [I[0][i], I[1][i]]

    # Conversion to sparse
    # Strangely slows down when there are several matrix multiplications
    # self.As = sparse.csr_matrix(self.As)
    # self.At = sparse.csr_matrix(self.At)

    # --- Other measurements

    if self.nE:
      
      # Connectivity
      self.is_strongly_connected = nx.is_strongly_connected(self.nx)

      # Diameter
      if self.is_strongly_connected:
        self.d = nx.diameter(self.nx)
      else:
        self.d = max([max(j.values()) for (i,j) in nx.shortest_path_length(self.nx)])

    else:

      self.d = 0
      
  # ========================================================================
  #                             MODIFICATIONS
  # ========================================================================

  def shuffle(self):

    # New network object
    Met = copy.deepcopy(self)

    # Shuffling indexes
    Icor = np.arange(self.nV)
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

  def complement(self):

    # New network object
    Met = Network(nNode=self.nV, directed=self.directed)

    # Adjacency matrix
    Met.Adj = np.logical_not(self.Adj)

    # Preparation
    Met.prepare()

    # --- Node attributes

    Met.node_attr = self.vrtx_attr

    # NB: No edge attribute when complementing.

    return Met

  # ========================================================================

  def degrade(self, type, delta, preserval=False, **kwargs):
    '''
    Network degradation

    Degradation can be done in many different ways ('type' argument):
    - Structure:
      'nd_rm', 'nr': Remove nodes (and the corresponding edges), equivalent to subgraph matching
      'ed_rm', 'er': Remove edges
      'ed_sw_src', 'es': Swap edges' sources
      'ed_sw_tgt', 'et': Swap edges' targets
      'ed_mv', 'em': Move edges, ie swap both sources and targets.

    - Attributes (to redo):
      'Cna': Change node attributes
      'Cea': Change edge attributes
      'Nna': add Gaussian noise to node attribute
      'Nea': add Gaussian noise to edge attribute

    + Can be at random (preserval=False) or in a given graph area (preserval=True)
    '''

    # New network object
    Net = copy.deepcopy(self)

    match type:

      case 'ed_rm' | 'er':

        # ------------------------------------------------------------------
        #     Remove edges
        # ------------------------------------------------------------------

        # Number of modifications
        nmod = round(delta*self.nE)
        
        if preserval:

          # Edge BFS with random node seed
          print(self.directed, self.nx.is_directed())
          Z = list(nx.edge_bfs(self.nx, source=0, orientation='ignore'))
          print(Z)

          # Indices

          pass

        else:

          # Indices
          I = np.ravel_multi_index(np.where(self.Adj), (self.nV, self.nV))
          J = np.random.choice(I, nmod, replace=False)
          K = np.unravel_index(J, (self.nV, self.nV))

          # Remove
          Net.Adj[K] = 0

      case 'Me':

        # ------------------------------------------------------------------
        #     Move edges
        # ------------------------------------------------------------------

        # Number of modifications
        nmod = round(delta*self.nE)

        # 0 → 1
        Ip = np.random.choice(np.ravel_multi_index(np.where(self.Adj==0), (self.nV, self.nV)), nmod, replace=False)
        Net.Adj[np.unravel_index(Ip,(self.nV, self.nV))] = 1

        # 1 → 0
        In = np.random.choice(np.ravel_multi_index(np.where(self.Adj==1), (self.nV, self.nV)), nmod, replace=False)
        Net.Adj[np.unravel_index(In,(self.nV, self.nV))] = 0
        
    # --- Output

    # Preparation
    Net.prepare(reset_edges=True)

    return Net

  # ========================================================================

  def subgraph(self, idx):

    # Create subnetwork
    Sub = type(self)()

    # Check
    if not isinstance(idx, list) and idx>self.nV:
      raise Exception(f"Subnetwork: The number of nodes in the subnet ({idx}) is greater than in the original network ({self.nV})")

    # Indexes
    I = idx if isinstance(idx, list) else np.random.choice(range(self.nV), idx, replace=False)
    # I = idx if isinstance(idx, list) else random.sample(range(self.nV), idx)
    K = np.ix_(I,I)

    # Adjacency matrix
    Sub.Adj = self.Adj[K]

    # --- Properties

    Sub.nNd = len(I)
    Sub.nEd = np.count_nonzero(Sub.Adj)
    Sub.nNa = self.nVa
    Sub.nEa = self.nEa

     # Preparation
    Sub.prepare()

    # --- Node attributes

    for attr in self.vrtx_attr:

      Sub.add_node_attr( {'measurable': attr['measurable'], 'values': attr['values'][I]} )
    
    # --- Edge attributes

    if self.nEa:

      # NB: Preparation has to be done before this point.

      # Compute indexes
      J = [np.where(np.all(self.edges==[I[e[0]], I[e[1]]], axis=1))[0][0] for e in Sub.edges]
      
      for attr in self.edge_attr:
        Sub.add_edge_attr( {'measurable': attr['measurable'], 'values': attr['values'][J]} )

    return Sub if isinstance(idx, list) else (Sub, I)
  
# ##########################################################################
#                        Graph generation functions
# ##########################################################################

# ------------------------------------------------------------------------
#                        Random graphs (Erdös-Rényi)
# ------------------------------------------------------------------------

def Gnm(n, m=None, p=None, a=None):
  '''
  G(n,m) or Erdös-Rényi random graph.
  In the ER model, the number of edges m is guaranteed.

  The parameter controlling the number of edges can be either:
  - The number of edges m (int)
  - The proportion of edges p (float, in [0,1])
  - The average number of edges per node a (float).
  '''

  # Number of edges
  if m is None:

    if p is not None:
      m = int(np.round(p*n**2))

    elif a is not None:
      m = int(np.round(a*n))

    else:
      raise Exception("The number of edges has to be defined with at least one of the parameters: 'm', 'p' or 'a'.") 

  # Network
  return Network(nx=nx.dense_gnm_random_graph(n, m, seed=np.random))

# ------------------------------------------------------------------------
#                           Star-branched-graph
# ------------------------------------------------------------------------

def star_branched(k, n, directed=False):
  '''
  Define the star-branched graph with k branches made of linear paths of length n.
  By default the graph is undirected.

  The total number of nodes is k*n+1.
  The central node index is 0.
  '''

  G = Network(k*n+1, directed=directed)
  G.Adj = np.full((G.nNd, G.nNd), False)

  z = 0
  for ki in range(k):
    z+=1
    G.Adj[0,z] = True
    for ni in range(n-1):
      G.Adj[z,z+1] = True
      z+=1
      
  G.prepare()

  return G
    