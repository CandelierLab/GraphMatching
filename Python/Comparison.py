import time
import warnings
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import time
import paprint as pa

from Matching import *

# === Comparison class =====================================================

class Comparison:

  # ========================================================================
  # |                                                                      |
  # |                          Constructor                                 |
  # |                                                                      |
  # ========================================================================

  def __init__(self, NetA, NetB, randomize_exploration=True, verbose=False,
               algorithm='GASM', eta=0.1):
    '''
    Comparison of two networks.

    The algorithm parameters can be:
    - 'Zager', as in [1]
    - 'GASM', Graph Attribute and Structure Matching (default)

    [1] L.A. Zager and G.C. Verghese, "Graph similarity scoring and matching",
        Applied Mathematics Letters 21 (2008) 86–94, doi: 10.1016/j.aml.2007.01.006
    '''

    # --- Definitions

    # The networks to compare
    self.NetA = NetA
    self.NetB = NetB

    # --- Algorithm and parameters

    # Algorithm
    self.algorithm = algorithm
    ''' The algorithm can be: "Zager", "GASM". '''

    self.randomize_exploration = randomize_exploration

    # Algorithm-dependant parameters
    match self.algorithm:

      case 'GASM':

        # Stochasticity factor
        self.eta = eta

    # --- Scores

    self.X = None
    self.Y = None

    # --- Misc

    self.verbose = verbose

  # ========================================================================
  # |                                                                      |
  # |                              Scores                                  |
  # |                                                                      |
  # ========================================================================

  def compute_scores(self, nIter=None, normalization=None,
            i_function=None, i_param={}, initial_evaluation=False, measure_time=False):
    ''' Score computation '''

    # --- Definitions --------------------------------------------------------

    # Number of nodes
    nA = self.NetA.nNd
    nB = self.NetB.nNd

    # Number of edges
    mA = self.NetA.nEd
    mB = self.NetB.nEd

    # --- Structural matching parameters

    if not mA or not mB:

      nIter = 0
      normalization = 1

    else:

      # Number of iterations
      if nIter is None:
        nIter = max(min(self.NetA.d, self.NetB.d), 1)

      # Normalization factor
      if normalization is None:
        normalization = 4*mA*mB/nA/nB + 1
    
    # --- Attributes ---------------------------------------------------------

    match self.algorithm:

      case 'Zager':

        # --- Node attributes

        # Base
        Xc = np.ones((nA,nB))

        for k, attr in enumerate(self.NetA.node_attr):

          bttr = self.NetB.node_attr[k]

          if attr['measurable']:
            pass
          else:
            # Build contraint attribute
            A = np.tile(attr['values'], (self.NetB.nNd,1)).transpose()
            B = np.tile(bttr['values'], (self.NetA.nNd,1))
            Xc *= A==B
        
        # Remapping in [-1, 1]
        Xc = Xc*2 - 1

      case 'GASM':

        # --- Node attributes
        
        if not nA or not nB:

          Xc = np.empty(0)
          Yc = np.empty(0)

        else:

          # Base
          Xc = np.ones((nA,nB))/normalization

          # Random initial fluctuations
          Xc += (np.random.rand(nA, nB)*2-1)*self.eta

          for k, attr in enumerate(self.NetA.node_attr):

            wA = attr['values']
            wB = self.NetB.node_attr[k]['values']

            if attr['measurable']:

              # --- Measurable attributes

              # Edge weights differences
              W = np.subtract.outer(wA, wB)

              sigma2 = np.var(W)
              if sigma2>0:
                Xc *= np.exp(-W**2/2/sigma2)

            else:

              # --- Categorical attributes

              Xc *= np.equal.outer(wA, wB)
              
          # --- Edge attributes

          # Base
          Yc = np.ones((mA,mB))

          if mA and mB:

            for k, attr in enumerate(self.NetA.edge_attr):

              wA = attr['values']
              wB = self.NetB.edge_attr[k]['values']

              if attr['measurable']:

                # --- Measurable attributes

                # Edge weights differences
                W = np.subtract.outer(wA, wB)

                sigma2 = np.var(W)
                if sigma2>0:
                  Yc *= np.exp(-W**2/2/sigma2)

              else:
                # --- Categorical attributes

                Yc *= np.equal.outer(wA, wB)

    # --- Computation --------------------------------------------------------

    # Iterative function settings
    if i_function is not None:
      i_param['NetA'] = self.NetA
      i_param['NetB'] = self.NetB
      output = []

    if not mA or not mB:

      self.X = Xc
      self.Y = Yc

    else:

      # Initialization
      self.X = np.ones((nA,nB))
      self.Y = np.ones((mA,mB))

      # Initial evaluation
      if i_function is not None and initial_evaluation:
        i_function(locals(), i_param, output)

      # --- Iterations
      
      for i in range(nIter):

        if measure_time:
          start = time.time()

        match self.algorithm:

          case 'Zager':
            self.X = self.NetA.As @ self.Y @ self.NetB.As.T + self.NetA.At @ self.Y @ self.NetB.At.T
            self.Y = self.NetA.As.T @ self.X @ self.NetB.As + self.NetA.At.T @ self.X @ self.NetB.At

            if normalization is None:
              # If normalization is not explicitely specified, the default normalization is used.
              self.X /= np.mean(self.X)
              self.Y /= np.mean(self.Y)
            else:
              self.X /= normalization

          case 'GASM':

            if i==0:

              self.X = (self.NetA.As @ self.Y @ self.NetB.As.T + self.NetA.At @ self.Y @ self.NetB.At.T + 1) * Xc
              self.Y = (self.NetA.As.T @ self.X @ self.NetB.As + self.NetA.At.T @ self.X @ self.NetB.At) * Yc

            else:

              self.X = (self.NetA.As @ self.Y @ self.NetB.As.T + self.NetA.At @ self.Y @ self.NetB.At.T + 1)
              self.Y = (self.NetA.As.T @ self.X @ self.NetB.As + self.NetA.At.T @ self.X @ self.NetB.At)
            
        ''' === A note on operation order ===

        If all values of X are equal to the same value x, then updating Y gives
        a homogeneous matrix with values 2x ; in this case the update of Y does
        not give any additional information. However, when all Y are equal the 
        update of X does not give an homogeneous matrix as some information 
        about the network strutures is included.

        So it is always preferable to start with the update of X.
        '''

        ''' === A note on normalization ===

        Normalization is useful for proving the convergence of the algorithm, 
        but is not necessary in the computation since the scores are relative
        and not absolute.
        
        So as long as the scores do not overflow the maximal float value, 
        there is no need to normalize. But this can happen quite fast.

        A good approximation of the normalization factor is:

                f = 2 sqrt[ (mA*mB)/(nA*nB) ] = 2 sqrt[ (mA/nA)(mB/nB) ]

        for an ER network with a density of edges p, we have m = p.n^2 so:
        
                f = 2 sqrt(nA.nB.pA.pB)

        Nevertheless, if one needs normalization as defined in Zager et.al.,
        it can be performed after the iterative procedure by dividing the final
        score matrices X and Y by:

                f = np.sqrt(np.sum(X**2))

        This is more efficient than computing the normalization at each
        iteration.
        '''

        if i_function is not None:
          i_function(locals(), i_param, output)

      # Final step
      if self.algorithm=='Zager':
        self.X = self.X * Xc

    # --- Output

    if i_function is not None:
      return output

  # ========================================================================
  # |                                                                      |
  # |                            Matching                                  |
  # |                                                                      |
  # ========================================================================

  def get_matching(self, force_perfect=True, **kwargs):
    ''' Compute one matching '''

    # --- Similarity scores --------------------------------------------------

    if self.X is None:

      if self.verbose:
        print('* No score matrix found, computing the score matrices.')
        start = time.time()

      if 'i_function' in kwargs:
        output = self.compute_scores(**kwargs)
      else:
        self.compute_scores(**kwargs)

      if self.verbose:
        print('* Scoring: {:.02f} ms'.format((time.time()-start)*1000))

    # --- Emptyness check ----------------------------------------------------

    if not self.X.size:
      return ([], output) if 'i_function' in kwargs else []

    # --- Solution search ---------------------------------------------------

    if self.verbose:
        tref = time.perf_counter_ns()

    # Prepare output
    M = Matching(self.NetA, self.NetB)

    # Jonker-Volgenant resolution of the LAP
    idxA, idxB = linear_sum_assignment(self.X, maximize=True)

    # match self.algorithm:

    #   case 'Zager' | 'SMOG':

    #     # Jonker-Volgenant resolution of the LAP
    #     idxA, idxB = linear_sum_assignment(self.X, maximize=True)

    #   case 'GASM':

    #     idxA, idxB = self.greedy_matching()

    #     # --- Forcing perfect matchings
        
    #     if force_perfect:

    #       # Perfect matching cardinality
    #       pmc = min(self.NetA.nNd, self.NetB.nNd)

    #       while len(idxA)<pmc:
            
    #         I = np.setdiff1d(np.arange(pmc), idxA)
    #         J = np.setdiff1d(np.arange(pmc), idxB)

    #         Ia, Jb = self.greedy_matching(I, J)

    #         # Concatenation
    #         idxA = np.concatenate((idxA, I[Ia]))
    #         idxB = np.concatenate((idxB, J[Jb]))

    #   case _:

    #     warnings.warn(f'Algorithm {self.algorithm} not found.')

    # --- Initialize matching object
        
    M.from_lists(idxA, idxB)
    M.compute_score(self.X)

    if self.verbose:
      print('* Matching: {:.02f} ms'.format((time.perf_counter_ns()-tref)*1e-6))

    # --- Output
    
    return (M, output) if 'i_function' in kwargs else M

  # === Greedy matching sub-method =========================================

  def greedy_matching(self, Isub=None, Jsub=None):

    # Subgraph delimitation
    if Isub is None:
      X = self.X
      AA = self.NetA.Adj
      AB = self.NetB.Adj

    else:
      X = self.X[np.ix_(Isub, Jsub)]
      AA = self.NetA.Adj[np.ix_(Isub, Jsub)]
      AB = self.NetB.Adj[np.ix_(Isub, Jsub)]

    # Matching indices
    idxA = []
    idxB = []

    # --- Initial matchup seed
    
    i0, j0 = get_max_from_array(X, self.randomize_exploration)
    idxA.append(i0)
    idxB.append(j0)

    # --- Greedy matching 

    while True:

      # Candidates
      cdd = []
      csc = []

      for k in range(len(idxA)):

        # Get all neighbors from both graphs not yet assigned
        if self.NetA.directed:

          I_in = np.setdiff1d(np.argwhere(AA[:,idxA[k]]).flatten(), idxA)
          I_out = np.setdiff1d(np.argwhere(AA[idxA[k],:]).flatten(), idxA)

          if not self.NetB.directed:
            I_ = np.concatenate((I_in, I_out))

        else:
          I_ = np.setdiff1d(np.argwhere(AA[idxA[k],:]).flatten(), idxA)

        if self.NetB.directed:

          J_in = np.setdiff1d(np.argwhere(AB[:,idxB[k]]).flatten(), idxB)
          J_out = np.setdiff1d(np.argwhere(AB[idxB[k],:]).flatten(), idxB)

          if not self.NetA.directed:
            J_ = np.concatenate((J_in, J_out))

        else:
          J_ = np.setdiff1d(np.argwhere(AB[idxB[k],:]).flatten(), idxB)

        if self.NetA.directed and self.NetB.directed:

          # Ingoing links
          if I_in.size and J_in.size:

            # Sub-matrix
            Sub = X[np.ix_(I_in, J_in)]
            i_, j_ = get_max_from_array(Sub, self.randomize_exploration)

            cdd.append([I_in[i_], J_in[j_]])
            csc.append(Sub[i_, j_])

          # Outgoing links
          if I_out.size and J_out.size:

            # Sub-matrix
            Sub = X[np.ix_(I_out, J_out)]
            i_, j_ = get_max_from_array(Sub, self.randomize_exploration)

            cdd.append([I_out[i_], J_out[j_]])
            csc.append(Sub[i_, j_])

        else:

          if I_.size and J_.size:

            # Sub-matrix
            Sub = X[np.ix_(I_, J_)]
            i_, j_ = get_max_from_array(Sub, self.randomize_exploration)

            cdd.append([I_[i_], J_[j_]])
            csc.append(Sub[i_, j_])

      if not len(csc):
        break

      k = get_max_from_array(csc, self.randomize_exploration)[0]

      idxA.append(cdd[k][0])
      idxB.append(cdd[k][1])

    return (idxA, idxB)

# === Usefull functions ====================================================
  
def get_max_from_array(X, randomize=True):

  if randomize:
    tmp = np.argwhere(X == np.max(X))
    ij = tmp[np.random.randint(tmp.shape[0])]    
  else:
    ij = np.unravel_index(np.argmax(X), X.shape)

  return ij

# === Matching OLD =========================================================

def matching_old(NetA, NetB, scores=None, threshold=None, all_solutions=True, max_solutions=None, structural_check=True, verbose=False, **kwargs):

  # --- Checks

  # No structural check for the Zager algorithm
  if 'algorithm' in kwargs and kwargs['algorithm']=='Zager':
    structural_check = False

  # --- Similarity scores

  if verbose:
    start = time.time()

  if scores is None:
    if 'i_function' in kwargs:
      (X, Y, output) = compute_scores(NetA, NetB, **kwargs)
    else:
      X = compute_scores(NetA, NetB, **kwargs)[0]
  else:
    X = scores

  if verbose:
    print('Scoring: {:.02f} ms'.format((time.time()-start)*1000), end=' - ')

  # --- Emptyness check

  if not X.size:
    return ([], output) if 'i_function' in kwargs else []

  # --- Threshold

  if threshold is not None:
    X[X<threshold] = -np.inf

  # --- Hungarian algorithm (Jonker-Volgenant)
    
  if verbose:
    tref = time.perf_counter_ns()

  I, J = linear_sum_assignment(X, maximize=True)

  if verbose:
    print('Matching: {:.02f} ms'.format((time.perf_counter_ns()-tref)*1e-6))

  # --- Output
  
  if not all_solutions:
    
    M = [[[I[k], J[k]] for k in range(len(I))]]

  else:

    # --- Total undetermination --------------------------------------------
    '''
    Sometimes the score matrix may be full of the same value. This can happend 
    when only one iteration is performed or if the graph is empty or full of 
    edges. This case result in a total indetermination and the number of 
    solutions rapidly explodes with min(nA,nB)! equivalent matchings.

    For computational reasons, in this case the returned MatchingSet object
    has an empty list of matchings and the flag property 'all_matchings' is
    set to True. The subsequent accuracy is then computed with the 
    theoretical value 1/min(nA,nB), not with the real list.
    '''

    if np.all(X==X[0]):
      
      # Display
      if verbose:
        print("Total undetermination: raising the 'all_matching' flag.")

      # Full matching set
      MS = MatchingSet(NetA, NetB, [], all_matchings=True)

      # Output
      if 'i_function' in kwargs:
        return (MS, output)
      else:
        return MS

    # --- Other cases (non-total undetermination) --------------------------

    # Solution score
    s = np.sum([X[I[k], J[k]] for k in range(len(I))])

    '''
    We follow the procedure described in:
      Finding All Minimum-Cost Perfect Matchings in Bipartite Graphs
      K. Fukuda and T. Matsui, NETWORKS Vol.22 (1992)
      https://doi.org/10.1002/net.3230220504

    with minor modifications to extend the algorithms to non square cost matrices.
    '''
    
    # --- Step 1: Admissible set -----------------------------------------

    # Square size
    nA = X.shape[0]
    nB = X.shape[1]

    # Mask & solution grid
    Mask = np.full((nA,nB), False)
    Grid = np.full((nA,nB), False)  
    for (i,j) in zip(I,J):
      Grid[i,j] = True
      
    # Display
    if verbose:
      print('')
      pa.matrix(X, highlight=Grid, title='Initial solution')

    # Minimal vectors
    mu = np.full(nA, -np.inf)
    mv = np.full(nB, -np.inf)

    # Maximal vectors
    Mu = np.full(nA, np.inf)
    Mv = np.full(nB, np.inf)

    # --- First step

    i0 = np.argmin(X[I,J])
    ref = [I[i0], J[i0]]
    Mask[I[i0], J[i0]] = True

    # Fix values
    mu[ref[0]] = Mu[ref[0]] = 0
    mv[ref[1]] = Mv[ref[1]] = X[ref[0],ref[1]]

    # --- Main loop 

    while True:

      # Min values    
      mu = np.maximum(mu, X[:,ref[1]] - mv[ref[1]])
      mv = np.maximum(mv, X[ref[0],:] - mu[ref[0]])

      # Max values    
      for (i,j) in zip(I,J):        
          Mu[i] = np.minimum(Mu[i], X[i,j] - mv[j])
          Mv[j] = np.minimum(Mv[j], X[i,j] - mu[i])

      # Max sums
      Muv = np.add.outer(Mu, Mv)

      # Stop condition
      Z = Muv - X + Mask

      # --- Debug display ------------
      # pa.line()
      # print(mu, mv, Mu, Mv)
      # pa.matrix(Muv, highlight=Mask)
      # pa.matrix(Z, highlight=Mask)
      # ------------------------------

      if np.isclose(np.min(Z), 0):

        # New reference
        tmp = np.where(Z==np.min(Z))
        ref = [tmp[0][0], tmp[1][0]]

      else:

        # Check the solution grid
        w = np.where(np.logical_and(Grid, np.logical_not(Mask)))

        if len(w[0]):

          mi = np.argmin(X[w[0], w[1]])

          # Set reference
          ref = [w[0][mi], w[1][mi]]

          # Update max vectors
          Mu[ref[0]] = mu[ref[0]]
          Mv[ref[1]] = X[ref[0],ref[1]] - mu[ref[0]]

        else:
          # If the solution grid is full: stop
          break

      # Update min vectors
      mu[ref[0]] = Mu[ref[0]]
      mv[ref[1]] = X[ref[0],ref[1]] - Mu[ref[0]]

      # Update mask
      Mask[ref[0], ref[1]] = True

    # Display
    if verbose:
      print('')
      pa.matrix(X, highlight=Mask, title='final')

    # --- Step 2: All perfect solutions of bipartite graph ---------------
    
    '''
    First, admissible set is decomposed in blocks.
    Then for each block all the solutions are computed. This step is faster 
    with the algorithm described in:
      Algorithms for enumerating all perfect, maximum and maximal matchings in bipartite graphs.
      Uno, T. Algorithms and Computation, Lecture Notes in Computer Science, vol 1350 (1997)
      https://doi.org/10.1007/3-540-63890-3_11

    For this we use the package py-bipartite-matching (0.2.0) available at:
      https://pypi.org/project/py-bipartite-matching/
    '''

    import py_bipartite_matching as pbm
    import networkx as nx

    # --- Bipartite graph

    # Initialization
    B = nx.Graph()

    # Nodes
    B.add_nodes_from(range(nA), bipartite=0)
    B.add_nodes_from(range(nA, nA+nB), bipartite=1)

    # Edges
    for (i,j) in zip(*np.nonzero(Mask)):
      B.add_edge(i, j+nA)

    # Find matchings and format output
    M = []

    for matching in pbm.enum_maximum_matchings(B):

      # Maximum number of solutions
      if max_solutions is not None:
        if len(M)==max_solutions: break

      # Format matching
      m = [[u, v-nA] for (u,v) in matching.items()]
      
      # Append without duplicates
      # if m not in M:    
      #   M.append(m)
      M.append(m)

    print('Number of solutions', len(M))

  # --- Step 3: Discard structurally unsound matchings ---------------------
  
  # Build matching set
  MS = MatchingSet(NetA, NetB, M)

  # --- Structural checks

  if structural_check:

    scorr = np.array([m.structural_correspondence for m in MS.matchings])
    I = np.where(scorr==np.max(scorr))[0]
    MS.matchings = [MS.matchings[i] for i in I]

    if verbose:
      print(f'Structural check: keeping {len(I)}/{len(M)} matchings.')

  # --- Output -------------------------------------------------------------

  if 'i_function' in kwargs:
    return (MS, output)
  else:
    return MS
