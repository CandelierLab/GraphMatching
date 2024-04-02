import copy
import numpy as np
from scipy import sparse
import networkx as nx

import paprint as pa

# ##########################################################################
#                          Generic Graph class
# ##########################################################################

class Graph:
  ''' Generic class for graphs '''
    
  # === CONSTRUCTOR ========================================================

  def __init__(self, nV=0, directed=True, Adj=None, nx=None):

    # Numbers
    self.nV = nV  # Number of verticex
    self.nVa = 0  # NUmber of vertex attributes
    self.nE = 0   # Number of edges 
    self.nEd = 0  # Number of directed edges (nE for directed graphs, 2*nE for undirected graphs)
    self.nEa = 0  # Number of edge attributes

    # Edges
    self.directed = directed
    self.edges = None

    # Adjacency matrix
    self.Adj = np.empty(0)

    # Attributes
    self.edge_attr = []
    self.vrtx_attr = []

    # Connected
    self.connected = None

    # Diameter
    self.d = None

    # --- Imports

    if nx is None:
      self.nx = None
      self.nxu = None
    else:
      self.import_from_networkx(nx)

    if Adj is not None:
      self.from_adjacency_matrix(Adj)

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
    Some info on the graph
    '''

    pa.line(self.__class__.__name__)

    s = f'\nDirected\n' if self.directed else f'\nUndirected\n'
    s += f'Number of vertices: {self.nV}\n'
    s += f'Number of edges: {self.nE}\n'

    return s
  
  def print(self, maxrow=20, maxcol=20):
    '''Extended info on the graph'''

    # Basic info
    print(self)

    # Adjacency matrix
    pa.matrix(self.Adj)

    # --- Graph properties

    if self.connected is not None:
      if self.directed:
        print('The graph is {:s}strongly connected'.format('' if self.connected else 'not '))
      else:
        print('The graph is {:s}connected'.format('' if self.connected else 'dis'))


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

  def from_adjacency_matrix(self, Adj):

    # Check adjacency matrix is symmetric for undirected graphs
    if not self.directed:
      Adj = np.logical_or(Adj, Adj.T)

    # Adjacency matrix
    self.Adj = Adj

    # Vertices
    self.nV = self.Adj.shape[0]

    # Edges
    self.nEd = np.count_nonzero(self.Adj)
    self.nE = self.nEd if self.directed else np.count_nonzero(np.triu(self.Adj))

    # Finalize preparation
    self.prepare()

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
    self.nEd = np.count_nonzero(self.Adj)

    # Preparation
    self.prepare()
    
  # ========================================================================
  #                             GENERATION
  # ========================================================================

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

  def add_vrtx_attr(self, *args, **kwargs):
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
    Prepares the Graph for comparison by:
    - Compute the graph connected state (strongly connected for directed graphs)
    - Computing the graph diameter
    - Computing the As and At matrices.

    Also: 
    - Establish the list of edges if it is empty
    - Create the corresponding networkx graphs if not present
    '''

    # --- Preparation

    # Edge list
    list_edges = self.edges is None

    if list_edges:
      self.edges = np.zeros((self.nE, 2), dtype=np.int32)

    # Source-edge and terminus-edge matrices
    self.As = np.zeros((self.nV, self.nEd), dtype=bool)
    self.At = np.zeros((self.nV, self.nEd), dtype=bool)

    # --- Loop through edges

    if self.directed:
      I = np.where(self.Adj)
    else:
      I = np.where(np.triu(self.Adj))

    for i in range(len(I[0])):

      self.As[I[0][i], i] = 1
      self.At[I[1][i], i] = 1

      if list_edges:
        self.edges[i,:] = [I[0][i], I[1][i]]

    # Conversion to Scipy sparse
    # Strangely slows down when there are several matrix multiplications
    # self.As = sparse.csr_matrix(self.As)
    # self.At = sparse.csr_matrix(self.At)

    # --- Networkx graphs
        
    if self.nx is None:

      if self.directed:
        self.nx = nx.from_numpy_array(self.Adj, create_using=nx.DiGraph)
        self.nxu = self.nx.to_undirected()
      else:
        self.nx = nx.from_numpy_array(self.Adj)
        self.nxu = self.nx

    # --- Other measurements

    if self.nE:
      
      # Connectivity
      if self.directed:
        self.connected = nx.is_strongly_connected(self.nx)
      else:
        self.connected = nx.is_connected(self.nx)

      # Diameter
      if self.connected:
        self.d = nx.diameter(self.nx)
      else:
        self.d = max([max(j.values()) for (i,j) in nx.shortest_path_length(self.nx)])

    else:

      self.d = 0
      
  # ========================================================================
  #                             MODIFICATIONS
  # ========================================================================

  def shuffle(self):
    '''
    Shuffled version of the graph, and shuffling indices.
    Use np.random for the RNG.
    '''

    # Shuffling indexes
    Idx = np.arange(self.nV)
    np.random.shuffle(Idx)

    # Shuffled graph 
    H = Graph(nV=self.nV, directed=self.directed, Adj=self.Adj[Idx, :][:, Idx])

    # --- Node attributes

    for a in self.vrtx_attr:

      attr = copy.deepcopy(a)
      attr['values'] = a['values'][Idx]
      H.add_vrtx_attr(attr)

    # --- Edge attributes

    if self.nEa:

      # Compute indexes
      J = [np.where(np.all(self.edges==[Idx[e[0]], Idx[e[1]]], axis=1))[0][0] for e in self.edges]
      
      for a in self.edge_attr:
        attr = copy.deepcopy(a)
        attr['values'] = a['values'][J]
        H.add_edge_attr(attr)

    return (H, Idx)

  # ========================================================================

  def complement(self):
    '''
    Complementary graph

    NB: No edge attribute can be kept when complementing.
    '''

    # --- Complement graph object
    
    H = Graph(nV=self.nV, directed=self.directed, Adj=np.logical_not(self.Adj))

    # --- Node attributes

    H.nVa = self.nVa
    H.vrtx_attr = self.vrtx_attr

    return H

  # ========================================================================

  def trim(self, Kv=None, Ke=None, Rv=None, Re=None):
    '''
    Trim nodes and edges.

    Parameters:
      Kv [int]: Vertices to keep
      Ke [int]: Edges to keep
      Rv [int]: Vertices to remove
      Re [int]: Edges to remove

    NB 1: If no parameteris defined, everything is kept and a copy of the graph is returned.
    NB 2: If vertices are removed, the associated edges are automatically added for removal.
    NB 3: If incoherent instructions are given, the precedence goes to:
      1) Edge removal due to vertice removal
      2) Keep instead of remove
    '''

    # --- Preparation ------------------------------------------------------

    # --- Vertices

    if Kv is None:
      Rv = np.empty(0) if Rv is None else np.array(Rv)
      Kv = np.setdiff1d(np.arange(self.nV), Rv)
    else:
      Rv = np.setdiff1d(np.arange(self.nV), Kv)

    # --- Associated edges to remove

    Ae = np.empty(0, dtype=int)
    for v in Rv:
      for i in np.where(self.Adj[v,:])[0]:
        if self.directed:
          Ae = np.concatenate((Ae, np.where(np.all(self.edges==[v,i], axis=1))[0]))
        else:
          if i>v:
            Ae = np.concatenate((Ae, np.where(np.all(self.edges==[v,i], axis=1))[0]))
          else:
            Ae = np.concatenate((Ae, np.where(np.all(self.edges==[i,v], axis=1))[0]))

    # --- Edges
      
    if Ke is None:
      Re = Ae if Re is None else np.union1d(np.array(Re, dtype=int), Ae)
    else:
      Re = np.union1d(np.setdiff1d(np.arange(self.nE), Ke), Ae)

    Ke = np.setdiff1d(np.arange(self.nE), Re)
    
    # --- Create degraded graph

    Adj = copy.deepcopy(self.Adj)

    # Remove edges
    if self.directed:
      Adj[self.edges[Re,0], self.edges[Re,1]] = False
    else:
      Adj[self.edges[Re,0], self.edges[Re,1]] = False
      Adj[self.edges[Re,1], self.edges[Re,0]] = False

    # Remove nodes
    Adj = Adj[Kv,:][:,Kv]

    H = Graph(Adj=Adj, directed=self.directed)
 
    # --- Attributes

    # Vertices attributes
    for attr in self.vrtx_attr:

      # Attribute reproduction
      a = {'measurable': attr['measurable'], 'values': attr['values'][Kv]}
      if 'name' in attr: a['name'] = attr['name']

      H.add_vrtx_attr(a)

    # Edge attributes
    for attr in self.edge_attr:

      # Attribute reproduction
      a = {'measurable': attr['measurable'], 'values': attr['values'][Ke]}
      if 'name' in attr: a['name'] = attr['name']

      H.add_edge_attr(a)

    return H

  # ========================================================================

  def degrade(self, type, delta, localization=False, source=None, **kwargs):
    '''
    Graph degradation

    Degradation can be done in many different ways ('type' argument):

    - Directivity (TO IMPLEMENT):
      'undirect': Undirect previsouly directed graphs
      'direct': Direct previsouly undirected graphs
      'reverse': Reverse the edges of a directed graph

    - Structure:
      'vx_rm', 'vr': Remove vertices (and the corresponding edges), equivalent to subgraph generation
      'ed_rm', 'er': Remove edges
      'ed_sw_src', 'es': Swap edges' sources
      'ed_sw_tgt', 'et': Swap edges' targets
      'ed_mv', 'em': Move edges, ie swap both sources and targets.

    - Attributes (TO IMPLEMENT):
      'Cna': Change node attributes
      'Cea': Change edge attributes
      'Nna': add Gaussian noise to node attribute
      'Nea': add Gaussian noise to edge attribute

    + Can be at random (localization=False) or in a given graph area. In the former case a breadth-first search is performed around a root node (source), and the algorithm can either preserve the surroundings of the root (localization='first') or remove it (localization='last').
    '''

    # Checks
    delta = min(max(delta,0),1)

    # Parameters of the new graph object
    nV = self.nV
    directed = self.directed
        
    match type:

      case 'vx_rm' | 'vr':

        # ------------------------------------------------------------------
        # Remove Vertices (and corresponding edges), equivalent to subgraph
        # ------------------------------------------------------------------

        H = self.subgraph(delta=delta, localization=localization)

      case 'ed_rm' | 'er':

        # ------------------------------------------------------------------
        #     Remove edges
        # ------------------------------------------------------------------

        # Number of modifications
        nmod = round(delta*self.nE)

        if localization:

          # Edge BFS with random node seed
          T = list(nx.edge_bfs(self.nx, source=None, orientation='ignore'))

          Re = []

          # Walk through BFS
          for i in range(nmod):

            # Degradation localization (first or last)
            match localization:
              case 'first': j = i
              case 'last': j = self.nE-i-1

            # Find edge index
            Re.append(np.where(np.logical_and(self.edges[:,0]==np.minimum(T[j][0], T[j][1]), self.edges[:,1]==np.maximum(T[j][0], T[j][1])))[0][0])          

        else:

          # Indices to remove
          Re = np.random.choice(self.nE, nmod, replace=False)

        # Degraded graph
        H = self.trim(Re=Re)
        
      case 'Me':

        # ------------------------------------------------------------------
        #     Move edges
        # ------------------------------------------------------------------

        pass

        # # Number of modifications
        # nmod = round(delta*self.nE)

        # # 0 → 1
        # Ip = np.random.choice(np.ravel_multi_index(np.where(self.Adj==0), (self.nV, self.nV)), nmod, replace=False)
        # H.Adj[np.unravel_index(Ip,(self.nV, self.nV))] = 1

        # # 1 → 0
        # In = np.random.choice(np.ravel_multi_index(np.where(self.Adj==1), (self.nV, self.nV)), nmod, replace=False)
        # H.Adj[np.unravel_index(In,(self.nV, self.nV))] = 0
        
    # --- Output

    return H

  # ========================================================================

  def subgraph(self, Idx=None, delta=None, localization=False):
    '''
    Subgraph generator.

    Idx can be:
      - int: the number of vertices to remove
      - list: list of vertices to remove
      - None: In this case the degradation parameter delta is used, with ('first', 'last') or without localization (False).
      
    NB: The associated edges are also removed.

    Return: the indices of kept vertices.
    '''

    # --- Checks

    if Idx is None and delta is None:
      raise Exception(f"Subgraph: either a number or list of vertices, or a node degradation ratio, has to be provided.")

    # --- Indexes to remove
    
    if Idx is not None:

      # Indices to remove
      Rv = Idx if isinstance(Idx, list) else np.random.choice(self.nV, Idx, replace=False)

    else:
    
      # Number of modifications
      nmod = round(delta*self.nV)

      if localization:

        # Edge BFS with random node seed
        T = list(nx.bfs_tree(self.nxu, source=np.random.randint(self.nV)))

        match localization:
          case 'first': Rv = T[:nmod]
          case 'last': Rv = T[nmod+1:]

      else:

        # Indices to remove
        Rv = np.random.choice(self.nV, nmod, replace=False)
      
    # Degraded graph
    H = self.trim(Rv=Rv)

    return H if isinstance(Idx, list) else (H, np.setdiff1d(np.arange(self.nV), Rv))
  
# ##########################################################################
#                        Graph generation functions
# ##########################################################################

# ------------------------------------------------------------------------
#                        Random graphs (Erdös-Rényi)
# ------------------------------------------------------------------------

def Gnm(n, m=None, p=None, a=None, directed=True, selfloops=True):
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

  # Check boundaries
  m = min(max(m, 0), n**2)

  # Selfloops
  if selfloops:

    if m==0:
      Adj = np.full((n,n), False)
    elif m==n**2:
      Adj = np.full((n,n), True)
    else:
      A = np.random.rand(n,n)
      Adj = A < np.sort(A.flatten())[p]

    # Output
    return Graph(nV=n, directed=directed, Adj=Adj)

  else:

    return Graph(nx=nx.gnm_random_graph(n, m, seed=np.random, directed=directed))

def Gnp(n, p=None, m=None, a=None, directed=True, selfloops=True):
  '''
  G(n,p) or Erdös-Rényi-Gilbert random graph.
  In the ERG model, the number of edges m is not guaranteed.

  The parameter controlling the number of edges can be either:
  - The number of edges m (int)
  - The proportion of edges p (float, in [0,1])
  - The average number of edges per node a (float).
  '''

  # Edge proportion
  if p is None:

    if m is not None:
      p = m/n**2
    elif a is not None:
      p = a/n
    else:
      raise Exception("The proportion of edges has to be defined with at least one of the parameters: 'p', 'm' or 'a'.") 

  # Check boundaries
  p = min(max(p, 0), 1)

  # Selfloops
  if selfloops:

    if p==0:
      Adj = np.full((n,n), False)
    elif p==1:
      Adj = np.full((n,n), True)
    else:
      Adj = np.random.rand(n,n) < p

    # Output
    return Graph(nV=n, directed=directed, Adj=Adj)

  else:

    return Graph(nx=nx.gnp_random_graph(n, p, seed=np.random, directed=directed))

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

  nV = k*n+1

  # --- Edges and adjacency matrix

  Adj = np.full((nV, nV), False)

  z = 0
  for ki in range(k):
    z+=1
    Adj[0,z] = True
    for ni in range(n-1):
      Adj[z,z+1] = True
      z+=1

  return Graph(nV, directed=directed, Adj=Adj)