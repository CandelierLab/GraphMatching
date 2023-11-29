import time
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy import sparse
import time
import paprint as pa

from Matching import *

# === Comparison ===========================================================

def compute_scores(NetA, NetB, nIter=None,
           algorithm='GASP', normalization=None,
           i_function=None, i_param={}, initial_evaluation=False, measure_time=False):
  '''
  Comparison of two networks.

  The algorithm parameters can be:
   - 'Zager', as in [1]
   - 'GASP' (default)

  [1] L.A. Zager and G.C. Verghese, "Graph similarity scoring and matching",
      Applied Mathematics Letters 21 (2008) 86–94, doi: 10.1016/j.aml.2007.01.006
  '''

  # --- Definitions --------------------------------------------------------

  # Number of nodes
  nA = NetA.nNd
  nB = NetB.nNd

  # Number of edges
  mA = NetA.nEd
  mB = NetB.nEd

  # --- Structural matching parameters

  if not mA or not mB:

    nIter = 0
    normalization = 1

  else:

    # Number of iterations
    if nIter is None:
      nIter = max(min(NetA.d, NetB.d), 1)

    # Normalization factor
    if normalization is None:
      normalization = 4*mA*mB/nA/nB + 1
  
  # --- Attributes ---------------------------------------------------------

  match algorithm:

    case 'Zager':

      # --- Node attributes

      # Base
      Xc = np.ones((nA,nB))

      for k, attr in enumerate(NetA.node_attr):

        bttr = NetB.node_attr[k]

        if attr['measurable']:
          pass
        else:
          # Build contraint attribute
          A = np.tile(attr['values'], (NetB.nNd,1)).transpose()
          B = np.tile(bttr['values'], (NetA.nNd,1))
          Xc *= A==B
      
      # Remapping in [-1, 1]
      Xc = Xc*2 - 1

    case 'GASP':

      # --- Node attributes
      
      if not nA or not nB:

        Xc = np.empty(0)
        Yc = np.empty(0)

      else:

        # Base
        Xc = np.ones((nA,nB))/normalization

        for k, attr in enumerate(NetA.node_attr):

          wA = attr['values']
          wB = NetB.node_attr[k]['values']

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

          for k, attr in enumerate(NetA.edge_attr):

            wA = attr['values']
            wB = NetB.edge_attr[k]['values']

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

  if not mA or not mB:

    X = Xc
    Y = Yc

  else:

    # Initialization
    X = np.ones((nA,nB))
    Y = np.ones((mA,mB))

    # Iterative function settings
    if i_function is not None:
      i_param['NetA'] = NetA
      i_param['NetB'] = NetB
      output = []

    # Initial evaluation
    if i_function is not None and initial_evaluation:
      i_function(locals(), i_param, output)

    # --- Iterations
    
    for i in range(nIter):

      if measure_time:
        start = time.time()

      match algorithm:

        case 'Zager':
          X = NetA.As @ Y @ NetB.As.T + NetA.At @ Y @ NetB.At.T
          Y = NetA.As.T @ X @ NetB.As + NetA.At.T @ X @ NetB.At

          if normalization is None:
            X /= np.mean(X)
            Y /= np.mean(Y)
          else:
            X /= normalization
      
        case 'GASP':

          if i==0:

            X = (NetA.As @ Y @ NetB.As.T + NetA.At @ Y @ NetB.At.T + 1) * Xc
            Y = (NetA.As.T @ X @ NetB.As + NetA.At.T @ X @ NetB.At) * Yc

          else:
          
            X = (NetA.As @ Y @ NetB.As.T + NetA.At @ Y @ NetB.At.T + 1)
            Y = (NetA.As.T @ X @ NetB.As + NetA.At.T @ X @ NetB.At)
            
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
    if algorithm=='Zager':
      X = X * Xc

  # --- Output

  if i_function is None:
    return(X, Y)
  else:
    return(X, Y, output)


# === Matching =============================================================
def matching(NetA, NetB, scores=None, threshold=None, all_solutions=True, max_solutions=None, structural_check=True, verbose=False, **kwargs):

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
    This step is faster with the algorithm described in:
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