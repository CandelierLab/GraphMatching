import time
import warnings
from collections import Counter
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

  def __init__(self, NetA, NetB, verbose=False):
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

  def compute_scores(self, algorithm='GASM',
            normalization=None,
            i_function=None, i_param={}, initial_evaluation=False, measure_time=False,
            **kwargs):
    ''' 
    Score computation 
    
    The algorithm can be: 'Zager', 'GASM'.

    'Zager' parameters:
      'nIter' (int): Number of iterations

    'GASM' parameters:
      'nIter' (int): Number of iterations
      'eta' (float): Noise level (default 1e-10)
    '''

    # --- Definitions --------------------------------------------------------

    GA = self.NetA
    GB = self.NetB

    ''' Check complements here '''

    # Number of nodes
    nA = GA.nNd
    nB = GB.nNd

    # Number of edges
    mA = GA.nEd
    mB = GB.nEd

    # --- Algorithms parameters

    # Number of iterations
    nIter = kwargs['nIter'] if 'nIter' in kwargs else max(min(GA.d, GB.d), 1)

    # Non-default normalization
    normalization = kwargs['normalization'] if 'normalization' in kwargs else None

    match algorithm:

      case 'Zager':

        # NB: normalization is performed dynamically during iterations.
        pass
        
      case 'GASM':
  
        # Normalization
        if normalization is None:

          # # Previous implementation
          # normalization = 4*mA*mB/nA/nB + 1

          dAi = np.sum(GA.Adj, axis=0)
          dAo = np.sum(GA.Adj, axis=1)
          dBi = np.sum(GB.Adj, axis=0)
          dBo = np.sum(GB.Adj, axis=1)

          normalization = np.outer(dAi,dBi) + np.outer(dAo,dBo)
          normalization[normalization==0] = 1

        # Noise
        eta = kwargs['eta'] if 'eta' in kwargs else 1e-10
            
    # --- Attributes ---------------------------------------------------------

    match algorithm:

      case 'Zager':

        # --- Node attributes

        # Base
        Xc = np.ones((nA,nB))
        
        for k, attr in enumerate(GA.node_attr):

          bttr = GB.node_attr[k]

          if attr['measurable']:
            pass
          else:
            # Build contraint attribute
            A = np.tile(attr['values'], (GB.nNd,1)).transpose()
            B = np.tile(bttr['values'], (GA.nNd,1))
            Xc *= A==B
        
        # Remapping in [-1, 1]
        Xc = Xc*2 - 1

        # --- Edge attributes
        
        Yc = np.ones((mA,mB))

      case 'GASM':

        # --- Node attributes
        
        if not nA or not nB:

          Xc = np.empty(0)
          Yc = np.empty(0)

        else:

          # Base
          # Xc = np.ones((nA,nB))/normalization
          Xc = np.ones((nA,nB))

          # Random initial fluctuations
          Xc += np.random.rand(nA, nB)*eta

          for k, attr in enumerate(GA.node_attr):

            wA = attr['values']
            wB = GB.node_attr[k]['values']

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

            for k, attr in enumerate(GA.edge_attr):

              wA = attr['values']
              wB = GB.edge_attr[k]['values']

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
      i_param['NetA'] = GA
      i_param['NetB'] = GB
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

        match algorithm:
        
          case 'Zager':

            self.X = GA.As @ self.Y @ GB.As.T + GA.At @ self.Y @ GB.At.T
            self.Y = GA.As.T @ self.X @ GB.As + GA.At.T @ self.X @ GB.At

            if normalization is None:
              # If normalization is not explicitely specified, the default normalization is used.
              self.X /= np.mean(self.X)
              self.Y /= np.mean(self.Y)
            
          case 'GASM':

            if i==0:
              self.X = (GA.As @ self.Y @ GB.As.T + GA.At @ self.Y @ GB.At.T + 1) * Xc
              self.Y = (GA.As.T @ self.X @ GB.As + GA.At.T @ self.X @ GB.At) * Yc

            else:
              self.X = (GA.As @ self.Y @ GB.As.T + GA.At @ self.Y @ GB.At.T + 1)
              self.Y = (GA.As.T @ self.X @ GB.As + GA.At.T @ self.X @ GB.At)

        # Normalization 
        if normalization is not None:
            self.X /= normalization

        if i_function is not None:
          i_function(locals(), i_param, output)

      # --- Post-processing
          
      match algorithm:
        
        case 'Zager':
          self.X = self.X * Xc

        case 'GASM':
          pass

    # --- Output

    if i_function is not None:
      return output

  # ========================================================================
  # |                                                                      |
  # |                            Matching                                  |
  # |                                                                      |
  # ========================================================================

  def get_matching(self, **kwargs):
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

    # --- Initialize matching object
        
    M.from_lists(idxA, idxB)
    M.compute_score(self.X)

    if self.verbose:
      print('* Matching: {:.02f} ms'.format((time.perf_counter_ns()-tref)*1e-6))

    # --- Output
    
    return (M, output) if 'i_function' in kwargs else M