import time
import pprint
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import itertools
import time
import copy
import paprint as pa

from ctypes import CDLL, POINTER
from ctypes import c_size_t, c_double

# load the library
GASP = CDLL("C/gasp.so")

# === Comparison ===========================================================

def scores(NetA, NetB, nIter=None,
           algorithm='GASP', normalization=None, language='Python',
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
            # *** Measurable attributes

            # Edge weights differences
            W = np.subtract.outer(wA, wB)

            sigma2 = np.var(W)
            if sigma2>0:
              Xc *= np.exp(-W**2/2/sigma2)

          else:
            # *** Non-measurable attributes

            tmp = np.equal.outer(wA, wB).astype(float)
            tmp[tmp==0] = 1/nA/nB
            
            # Xc *= np.equal.outer(wA, wB)
            Xc *= tmp
        
        # --- Edge attributes

        # Base
        Yc = np.ones((mA,mB))

        if mA and mB:

          for k, attr in enumerate(NetA.edge_attr):

            wA = attr['values']
            wB = NetB.edge_attr[k]['values']

            if attr['measurable']:
              # *** Measurable attributes

              # Edge weights differences
              W = np.subtract.outer(wA, wB)

              sigma2 = np.var(W)
              if sigma2>0:
                Yc *= np.exp(-W**2/2/sigma2)

            else:
              # *** Non-measurable attributes

              Yc *= np.equal.outer(wA, wB)


  # --- Computation --------------------------------------------------------

  if not mA or not mB:

    X = Xc
    Y = Yc

  else:

    # Preallocation
    X = np.ones((nA,nB))
    Y = np.ones((mA,mB))

    if i_function is not None:
      i_param['NetA'] = NetA
      i_param['NetB'] = NetB
      output = []

    # Initial evaluation
    if i_function is not None and initial_evaluation:
      i_function(locals(), i_param, output)

    match language:

      case 'C':

        # NB: Zager is not supported yet with the C++ implementation

        # Prototypes
        p_np_float = np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C")
        p_np_int = np.ctypeslib.ndpointer(dtype=np.int32, ndim=2, flags="C")
        GASP.scores.argtypes = [p_np_float, p_np_float, p_np_int, p_np_int, c_size_t, c_size_t, c_size_t, c_size_t, c_double]
        GASP.scores.restype = None

        # Compute scores
        GASP.scores(X, Y, NetA.edges, NetB.edges, nA, nB, mA, mB, normalization, nIter)

      case 'Python':

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
              X = (NetA.As @ Y @ NetB.As.T + NetA.At @ Y @ NetB.At.T +1) * Xc
              Y = (NetA.As.T @ X @ NetB.As + NetA.At.T @ X @ NetB.At) * Yc

            case 'GASP2':
              X = (1 + NetA.As @ Y @ NetB.As.T + NetA.At @ Y @ NetB.At.T) * Xc
              Y = (1 + NetA.As.T @ X @ NetB.As + NetA.At.T @ X @ NetB.At) * Yc

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

          This is much more efficient than computing the normalization at each
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
def matching(NetA, NetB, threshold=None, all_solutions=True, brute=False, verbose=False, **kwargs):

  # --- Similarity scores

  if verbose:
    start = time.time()

  if 'i_function' in kwargs:
    (X, Y, output) = scores(NetA, NetB, **kwargs)
  else:
    X = scores(NetA, NetB, **kwargs)[0]

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
    start = time.time()

  I, J = linear_sum_assignment(X, maximize=True)

  if verbose:
    print('Matching: {:.02f} ms'.format((time.time()-start)*1000))

  # --- Output

  if all_solutions:

    # Solution score
    s = np.sum([X[I[k], J[k]] for k in range(len(I))])

    if brute:

      # Candidates
      C = [[[I[k], J[k]] for k in range(len(I))]]

      # Solution container
      M = []

      # Loop over candidates
      while C:

        # Append solution
        m = C.pop()
        M.append(m)

        # Test all possible dual inversions
        for d in itertools.combinations(range(len(m)), 2):
          
          if X[m[d[0]][0], m[d[0]][1]]+X[m[d[1]][0], m[d[1]][1]] == X[m[d[0]][0], m[d[1]][1]]+X[m[d[1]][0], m[d[0]][1]]:
            
            # New solution
            ns = copy.deepcopy(m)
            ns[d[1]][1] = m[d[0]][1]
            ns[d[0]][1] = m[d[1]][1]

            if ns not in M and ns not in C:
              C.append(ns)

    else:

      # Closeness parameters
      rtol = 1e-10
      atol = 1e-8

      # --- Pad with zeros (nA != nB)

      if X.shape[0] > X.shape[1]:
        A0 = []
        A1 = np.arange(X.shape[1], X.shape[0])
        X = np.append(X, np.zeros((X.shape[0], A1.size)), axis=1)

      elif X.shape[0] < X.shape[1]:
        A0 = np.arange(X.shape[0], X.shape[1])
        A1 = []
        X = np.append(X, np.zeros((A0.size, X.shape[1])), axis=0)

      else:
        A0 = []
        A1 = []
        
      # --- Preparation

      pa.matrix(X, title='X initial')

      # Initial first line
      X0 = X[0,:]

      # Remove solution
      for (i,j) in zip(I,J):
        X[:,j] -= X[i,j]

      pa.matrix(X, title='X prepared')

      # --- Correction loop

      for i in range(X.shape[0]):

        # Bipartite graph weights
        U = np.max(X, axis=1)
        V = X0 - X[0,:]

        # Current score
        c = np.sum(U)+np.sum(V)

        print(f'Diff: {s-c}')

        # Termination check
        if np.isclose(c, s, rtol=rtol, atol=atol):
          break

        L = X[i,:].copy()

        # Where to remove ?
        mi = np.argmax(L)
        m = L[mi]

        # How much to remove?
        d = m-np.max(np.delete(L, mi))

        # Remove
        X[:,mi] -= d

        print(f'Removing {d:.3f} from column {mi}.')
        pa.matrix(X)


      M = []

      
  else:
    M = [(I[k], J[k]) for k in range(len(I))]

  if 'i_function' in kwargs:
    return (M, output)
  else:
    return M
  
# ----------------------------------------------------------------------
def allsolutions(S, I, J):

  pa.matrix(S)

  # --- Preparation

  n = len(I)

  # Mask & score
  s = 0
  Mask = np.full((n,n), True)  
  for (i,j) in zip(I,J):
    s += S[i,j]
    Mask[i,j] = False
    
  # Minimal vectors
  mu = np.full(n, -np.inf)
  mv = np.full(n, -np.inf)

  # Maximal vectors
  Mu = np.full(n, np.inf)
  Mv = np.full(n, np.inf)

  # Fixed value
  mu[I[0]] = 0  
  Mu[J[0]] = 0

  ref = [I[0], J[0]]

  # Fix value
  mv[ref[1]] = S[ref[0],ref[1]] - mu[ref[0]]
  Mv[ref[1]] = S[ref[0],ref[1]] - Mu[ref[0]]

  for iter in range(4):

    # Min values    
    mu = np.maximum(mu, S[:,ref[1]] - mv[ref[1]])
    mv = np.maximum(mv, S[ref[0],:] - mu[ref[0]])

    # Max values    
    for (i,j) in zip(I,J):
      Mu[i] = S[i,j] - mv[i]
      Mv[j] = S[i,j] - mu[i]

    # Max sums
    Muv = np.add.outer(Mu, Mv)

    print('')
    pa.line(f'Iteration {iter}')
    print('Minimal:')
    print(np.stack((mu, mv), axis=-1))  
    print('Maximal:')
    print(np.stack((Mu, Mv), axis=-1)) 
    pa.matrix(Muv)

    # New reference
    tmp = np.nonzero(np.logical_and(Muv==S, Mask))
    if not tmp[0].size:
      assert s == np.sum(mu) + np.sum(mv)
      break

    ref = [tmp[0][0], tmp[1][0]]


    print('Reference: ', ref)

    # --- Updates

    # Min vectors
    mu[ref[0]] = Mu[ref[0]]
    mv[ref[1]] = Mv[ref[1]]

    # Mask
    Mask[ref[0], ref[1]] = False

  pa.matrix(np.logical_not(Mask))

    