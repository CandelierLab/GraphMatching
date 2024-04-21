import time
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment, quadratic_assignment
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

  def __init__(self, Ga, Gb, verbose=False):
    '''
    Comparison of two graphs.

    The algorithm parameters can be:
    - 'FAQ', as in [1]
    - 'Zager', as in [2]
    - 'GASM', Graph Attribute and Structure Matching (default)

    [1] J.T. Vogelstein et al., "Fast Approximate Quadratic Programming for Graph Matching",
      PLoS One 10(4) (2015); doi:10.1371/journal.pone.0121002

    [2] L.A. Zager and G.C. Verghese, "Graph similarity scoring and matching",
        Applied Mathematics Letters 21 (2008) 86–94; doi: 10.1016/j.aml.2007.01.006
    '''

    # --- Definitions

    # The networks to compare
    self.Ga = Ga
    self.Gb = Gb

    # --- Scores

    self.X = None
    self.Y = None

    # --- Misc

    self.info = {}
    self.verbose = verbose

  # ========================================================================
  # |                                                                      |
  # |                              Scores                                  |
  # |                                                                      |
  # ========================================================================

  def compute_scores(self, algorithm='GASM', **kwargs):
    ''' 
    Score computation 
    
    The algorithm can be: 'Zager', 'GASM'.

    'Zager' parameters:
      'nIter' (int): Number of iterations
      'normalization' (float or np.Array): normalization factor(s)

    'GASM' parameters:
      'nIter' (int): Number of iterations
      'normalization' (float or np.Array): normalization factor(s)
      'eta' (float): Noise level (default 1e-10)
    '''

    # --- Definitions --------------------------------------------------------

    Ga = self.Ga
    Gb = self.Gb

    # Complement
    if algorithm=='GASM':

      if Ga.directed:
        complement = Ga.nE + Gb.nE > (Ga.nV**2 + Gb.nV**2)/2
      else:
        complement = Ga.nE + Gb.nE > (Ga.nV*(Ga.nV+1) + Gb.nV*(Gb.nV+1))/4

      if complement:
        Ga = self.Ga.complement()
        Gb = self.Gb.complement()

    # Number of vertices
    nA = Ga.nV
    nB = Gb.nV

    # Number of edges    
    mA = Ga.nE
    mB = Gb.nE

    # --- Algorithms parameters

    # Number of iterations
    nIter = kwargs['nIter'] if 'nIter' in kwargs else max(min(Ga.d, Gb.d), 1)
    self.info['nIter'] = nIter

    # Non-default normalization
    normalization = kwargs['normalization'] if 'normalization' in kwargs else None

    match algorithm:

      case 'Zager':

        ''' Normalization is performed dynamically during iterations. '''
        pass
        
      case 'GASM':
  
        # Normalization
        if normalization is None:
          normalization = 4*mA*mB/nA/nB if nA and nB else 1
          # normalization = 4*mA*mB/nA/nB + 1 if nA and nB else 1

        # Noise
        eta = kwargs['eta'] if 'eta' in kwargs else 1e-10
            
    # --- Attributes ---------------------------------------------------------

    match algorithm:

      case 'Zager':

        # --- Node attributes

        # Base
        N = np.ones((nA,nB))
        
        for k, attr in enumerate(Ga.vrtx_attr):

          bttr = Gb.vrtx_attr[k]

          if attr['measurable']:
            pass
          else:
            # Build contraint attribute
            A = np.tile(attr['values'], (nB,1)).transpose()
            B = np.tile(bttr['values'], (nA,1))
            N *= A==B
        
        # Remapping in [-1, 1]
        N = N*2 - 1

        # --- Edge attributes
        
        E = np.ones((mA,mB))

      case 'GASM':

        if not nA or not nB:

          N = np.empty(0)
          E = np.empty(0)

        else:

          # --- Node attributes

          # Base
          N = np.ones((nA,nB))

          for k, attr in enumerate(Ga.vrtx_attr):

            wA = attr['values']
            wB = Gb.vrtx_attr[k]['values']

            if attr['measurable']:

              # --- Measurable attributes

              # Edge weights differences
              W = np.subtract.outer(wA, wB)

              sigma2 = np.var(W)
              if sigma2>0:
                N *= np.exp(-W**2/2/sigma2)

            else:

              # --- Categorical attributes

              N *= np.equal.outer(wA, wB)

          # --- Edge attributes

          # Base
          E = np.ones((mA,mB))

          if mA and mB:

            for k, attr in enumerate(self.Ga.edge_attr):

              wA = attr['values']
              wB = self.Gb.edge_attr[k]['values']

              if attr['measurable']:

                # --- Measurable attributes

                # Edge weights differences
                W = np.subtract.outer(wA, wB)

                E = W==0
                
                # sigma2 = np.var(W)
                # if sigma2>0:
                #   E *= np.exp(-W**2/2/sigma2)

              else:
                # --- Categorical attributes

                E *= np.equal.outer(wA, wB)

        # Random initial fluctuations
        H = np.random.rand(nA, nB)*eta

    # --- Computation --------------------------------------------------------
    
    # pa.matrix(N)
    pa.matrix(E, title='E', maxrow=100)

    if not mA or not mB:

      self.X = N
      self.Y = E

    else:

      # --- Initialization

      if not nIter: self.X = np.ones((nA, nB))  
      self.Y = np.ones((mA, mB))

      # --- Iterations

      if self.verbose:
          t0 = time.perf_counter_ns()

      for i in range(nIter):

        if self.verbose:
          ti = time.perf_counter_ns()

        ''' === A note on operation order ===

        If all values of X are equal to the same value x, then updating Y gives
        a homogeneous matrix with values 2x ; in this case the update of Y does
        not give any additional information. However, when all Y are equal the 
        update of X does not give an homogeneous matrix as some information 
        about the network strutures is included.

        + Network complement

        So it is always preferable to start with the update of X.
        '''

        ''' === A note on normalization ===

        Normalization is useful for proving the convergence of the algorithm, but is not necessary in the computation since the scores are relative
        and not absolute.
        
        So as long as the scores do not overflow the maximal float value, there is no need to normalize. But this can happen quite fast.

        A good approximation of the normalization factor is:

                f = 2 sqrt[ (mA*mB)/(nA*nB) ] = 2 sqrt[ (mA/nA)(mB/nB) ]

        for an ER network with a density of edges p, we have m = p.n^2 so:
        
                f = 2 sqrt(nA.nB.pA.pB)

        Nevertheless, if one needs normalization as defined in Zager et.al., it can also be performed after the iterative procedure by dividing the final score matrices X and Y by:

                f = np.sqrt(np.sum(X**2))

        This is slightly more efficient than computing the normalization at each iteration.
        '''

        match algorithm:
        
          case 'Zager':

            if Ga.directed:
              self.X = Ga.S @ self.Y @ Gb.S.T + Ga.T @ self.Y @ Gb.T.T
              self.Y = Ga.S.T @ self.X @ Gb.S + Ga.T.T @ self.X @ Gb.T

            else:
              self.X = Ga.R @ self.Y @ Gb.R.T
              self.Y = Ga.R.T @ self.X @ Gb.R

            if normalization is None:
              # If normalization is not explicitely specified, the default normalization is used.
              self.X /= np.mean(self.X)
              self.Y /= np.mean(self.Y)
            
          case 'GASM':

            if Ga.directed:

              if i==0:
                X0 = (self.Ga.S @ E @ self.Gb.S.T + self.Ga.T @ E @ self.Gb.T.T) * (N+H)
                self.Y = Ga.S.T @ X0 @ Gb.S + Ga.T.T @ X0 @ Gb.T
              else:
                self.Y = Ga.S.T @ self.X @ Gb.S + Ga.T.T @ self.X @ Gb.T

              self.X = (Ga.S @ self.Y @ Gb.S.T + Ga.T @ self.Y @ Gb.T.T)

              pa.line(str(i))
              if i==0:
                pa.matrix(X0, maxrow=100)
              pa.matrix(self.Y, maxrow=100)
              pa.matrix(self.X, maxrow=100)

            else:

              if i==0:
                X0 = (self.Ga.R @ E @ self.Gb.R.T) * (N+H)
                self.Y = Ga.R.T @ X0 @ Gb.R
              else:
                self.Y = Ga.R.T @ self.X @ Gb.R

              self.X = Ga.R @ self.Y @ Gb.R.T

        # --- Normalization 
              
        if normalization is not None:
            self.Y /= normalization

        # --- Information
            
        if 'info_avgScores' in kwargs:

          # Initialization
          if 'avgX' not in self.info:
            self.info['avgX'] = []

          # Update
          self.info['avgX'].append(np.mean(self.X))

      # --- Timing
          
      if self.verbose:
        print('Total Iterations', (time.perf_counter_ns()-t0)*1e-6, 'ms')

      # --- Post-processing
          
      match algorithm:
        
        case 'Zager':

          self.X = self.X * N

        case 'GASM':
          
          # Isolated vertices
          I = self.X==0
          self.X[I] = N[I]

  # ========================================================================
  # |                                                                      |
  # |                            Matching                                  |
  # |                                                                      |
  # ========================================================================

  def get_matching(self, algorithm='GASM', **kwargs):
    ''' Compute one matching '''

    # Prepare output
    M = Matching(self.Ga, self.Gb, algorithm=algorithm)
    M.time = {'total': None}

    # Measure time
    tref = time.perf_counter_ns()

    match algorithm:

      case 'random':

        # Populate the matching object
        Idx = np.arange(self.Ga.nV)
        np.random.shuffle(Idx)
        M.from_lists(np.arange(self.Ga.nV), Idx)

        M.time['total'] = (time.perf_counter_ns()-tref)*1e-6

      case 'FAQ':

        # Solve the Quadratic Assignment Problem

        # Check weight presence
        if self.Ga.nEa==0:
          A = self.Ga.Adj
          B = self.Gb.Adj
        else:

          A = np.zeros((self.Ga.nV, self.Ga.nV), dtype=float)
          for i, e in enumerate(self.Ga.edges):
            A[e[0], e[1]] = self.Ga.edge_attr[0]['values'][i]
            if not self.Ga.directed:
              A[e[1], e[0]] = A[e[0], e[1]]

          B = np.zeros((self.Gb.nV, self.Gb.nV), dtype=float)
          for i, e in enumerate(self.Gb.edges):
            B[e[0], e[1]] = self.Gb.edge_attr[0]['values'][i]
            if not self.Gb.directed:
              B[e[1], e[0]] = B[e[0], e[1]]

        res = quadratic_assignment(A, B, options={'maximize': True})
        
        # Record computing time
        M.time['total'] = (time.perf_counter_ns()-tref)*1e-6

        # Populate the matching object
        M.from_lists(np.arange(self.Ga.nV), res.col_ind)
        M.score = res.fun

      case 'Zager' | 'GASM':

        # --- Similarity scores --------------------------------------------------

        force = kwargs['force'] if 'force' in kwargs else False

        if self.X is None or force:

          if self.verbose:
            print('* No score matrix found, computing the score matrices.')

          if 'i_function' in kwargs:
            output = self.compute_scores(algorithm=algorithm, **kwargs)
          else:
            self.compute_scores(algorithm=algorithm, **kwargs)

          # Informations
          M.info = self.info
          
          if self.verbose:
            print('* Scoring: {:.02f} ms'.format((time.time()-tref)*1000))

        M.time['scores'] = (time.perf_counter_ns()-tref)*1e-6
        tref = time.perf_counter_ns()
        
        # --- Emptyness check ----------------------------------------------

        if not self.X.size:
          M.time['LAP'] = (time.perf_counter_ns()-tref)*1e-6
          M.time['total'] = (time.perf_counter_ns()-tref)*1e-6
          M.initialize()
          return M

        # --- Solution search ----------------------------------------------

        # Jonker-Volgenant resolution of the LAP
        idxA, idxB = linear_sum_assignment(self.X, maximize=True)

        # Record computing time
        M.time['LAP'] = (time.perf_counter_ns()-tref)*1e-6
        M.time['total'] = M.time['scores'] + M.time['LAP']

        # --- Initialize matching object
            
        M.from_lists(idxA, idxB)
        M.compute_score(self.X)

    # --- Output
    
    return M


