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

  def __init__(self, NetA, NetB, randomize_exploration=True, verbose=False,
               algorithm='GASM', eta=1e-10):
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
            i_function=None, i_param={}, initial_evaluation=False, measure_time=False, deg_norm=False):
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

        # --- Edge attributes
        
        Yc = np.ones((mA,mB))

      case 'GASM':

        # --- Node attributes
        
        if not nA or not nB:

          Xc = np.empty(0)
          Yc = np.empty(0)

        else:

          # Base
          Xc = np.ones((nA,nB))/normalization

          # Random initial fluctuations
          Xc += np.random.rand(nA, nB)*self.eta

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

      self.X = np.real(Xc)
      self.Y = Yc

    else:

      if deg_norm:
        dAi = np.sum(self.NetA.Adj, axis=0)
        dAo = np.sum(self.NetA.Adj, axis=1)
        dBi = np.sum(self.NetB.Adj, axis=0)
        dBo = np.sum(self.NetB.Adj, axis=1)

        dAB = np.outer(dAi,dBi) + np.outer(dAo,dBo)

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

        match self.algorithm:
        
          case 'Zager':

            self.X = self.NetA.As @ self.Y @ self.NetB.As.T + self.NetA.At @ self.Y @ self.NetB.At.T
            self.Y = self.NetA.As.T @ self.X @ self.NetB.As + self.NetA.At.T @ self.X @ self.NetB.At

            if normalization is None:
              # If normalization is not explicitely specified, the default normalization is used.
              self.X /= np.mean(self.X)
              self.Y /= np.mean(self.Y)
            
          case 'GASM':

            if i==0:

              self.X = (self.NetA.As @ self.Y @ self.NetB.As.T + self.NetA.At @ self.Y @ self.NetB.At.T + 1) * Xc
              self.Y = (self.NetA.As.T @ self.X @ self.NetB.As + self.NetA.At.T @ self.X @ self.NetB.At) * Yc

            else:

              self.X = (self.NetA.As @ self.Y @ self.NetB.As.T + self.NetA.At @ self.Y @ self.NetB.At.T + 1)
              self.Y = (self.NetA.As.T @ self.X @ self.NetB.As + self.NetA.At.T @ self.X @ self.NetB.At)

        # Normalization 
        if normalization is not None and normalization!=1:
            self.X /= normalization
          
        if deg_norm:            
            self.X /= dAB

        if i_function is not None:
          i_function(locals(), i_param, output)

      # --- Post-processing
          
      match self.algorithm:
        
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

    # --- Initialize matching object
        
    M.from_lists(idxA, idxB)
    M.compute_score(self.X)

    if self.verbose:
      print('* Matching: {:.02f} ms'.format((time.perf_counter_ns()-tref)*1e-6))

    # --- Output
    
    return (M, output) if 'i_function' in kwargs else M