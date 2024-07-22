import time
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment, quadratic_assignment
import time
import numba as nb
import numba.cuda as cuda

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
    - random, (fast but not very efficient)
    - 'FAQ', as in [1]
    - '2opt', as in [2]
    - 'Zager', as in [3]
    - 'GASM', Graph Attribute and Structure Matching (default)

    [1] J.T. Vogelstein et al., "Fast Approximate Quadratic Programming for Graph Matching",
      PLoS One 10(4) (2015); doi:10.1371/journal.pone.0121002

    [2] D. E. Fishkind et al., "See de d graph matching", Pattern recognition 87, 203-215 (2019); doi:10.1016/j.patcog.2018.09.014

    [3] L.A. Zager and G.C. Verghese, "Graph similarity scoring and matching",
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

    match algorithm.lower():

      case 'random':

        # Populate the matching object
        Idx = np.arange(self.Ga.nV)
        np.random.shuffle(Idx)
        M.from_lists(np.arange(self.Ga.nV), Idx)

        M.time['total'] = (time.perf_counter_ns()-tref)*1e-6

      case 'faq' | '2opt':

        # Solve the Quadratic Assignment Problem

        # Check attributes
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

        res = quadratic_assignment(A, B, method=algorithm.lower(), options={'maximize': True})
        
        # Record computing time
        M.time['total'] = (time.perf_counter_ns()-tref)*1e-6

        # Populate the matching object
        M.from_lists(np.arange(self.Ga.nV), res.col_ind)
        M.score = res.fun

      case 'zager' | 'gasm':

        # --- Similarity scores --------------------------------------------------

        force = kwargs['force'] if 'force' in kwargs else False

        if self.X is None or force:

          if self.verbose:
            print('* No score matrix found, computing the score matrices.')

          # if 'i_function' in kwargs:
          #   output = self.compute_scores(algorithm=algorithm, **kwargs)
          # else:

          match algorithm.lower():

            case 'zager':
              self.compute_scores_Zager(**kwargs)

            case 'gasm':

              self.compute_scores_GASM(**kwargs)

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

      if 'complement' in kwargs:
        complement  = kwargs['complement']
        
      else:
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

          if  attr['precision'] is None or attr['precision']:

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

            if attr['precision'] is None or attr['precision']:

              # --- Measurable attributes

              # Edge weights differences
              W = np.subtract.outer(wA, wB)

              if attr['precision'] is None:
                rho_a2 = np.var(W)
              else:
                rho_a2 = attr['precision']**2

              if rho_a2>0:
                N *= np.exp(-W**2/2/rho_a2)

            else:

              # --- Categorical attributes

              N *= np.equal.outer(wA, wB)

          # --- Edge attributes

          # Base
          E = np.ones((self.Ga.nE, self.Gb.nE))

          if self.Ga.nE and self.Gb.nE:

            for k, attr in enumerate(self.Ga.edge_attr):

              wA = attr['values']
              wB = self.Gb.edge_attr[k]['values']

              if attr['precision'] is None or attr['precision']:

                # --- Measurable attributes

                # Edge weights differences
                W = np.subtract.outer(wA, wB)

                if attr['precision'] is None:
                  rho_a2 = np.var(W)
                else:
                  rho_a2 = attr['precision']**2

                if rho_a2>0:
                  E *= np.exp(-W**2/2/rho_a2)

              else:
                # --- Categorical attributes

                E *= np.equal.outer(wA, wB)

        # Random initial fluctuations
        H = np.random.rand(nA, nB)*eta

    # --- Computation --------------------------------------------------------
    
    # pa.matrix(N)
    # pa.matrix(E, title='E', maxrow=100)

    if not mA or not mB:

      self.X = N
      self.Y = E

    else:

      # --- Initialization
        
      match algorithm:
      
        case 'GASM':

          # Define X0
          if Ga.directed:
            X0 = (self.Ga.S @ E @ self.Gb.S.T + self.Ga.T @ E @ self.Gb.T.T) * (N+H)
          else:
            X0 = (self.Ga.R @ E @ self.Gb.R.T) * (N+H)

          # pa.matrix(X0, title='X0', maxrow=100, highlight=X0>0.5)

          if not nIter:
            self.X = X0

        case _:
          
          if not nIter:
            self.X = np.ones((nA, nB))  

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
                self.Y = Ga.S.T @ X0 @ Gb.S + Ga.T.T @ X0 @ Gb.T
              else:
                self.Y = Ga.S.T @ self.X @ Gb.S + Ga.T.T @ self.X @ Gb.T

              self.X = (Ga.S @ self.Y @ Gb.S.T + Ga.T @ self.Y @ Gb.T.T)

              # pa.line(str(i))
              # # pa.matrix(self.Y, maxrow=100)
              # pa.matrix(self.X, maxrow=100, highlight=X0>0.5)

            else:

              if i==0:
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

  def compute_scores_Zager(self, **kwargs):
    ''' 
    Score computation with Zager's algorithm
    
    Parameters:
      'nIter' (int): Number of iterations
      'normalization' (float or np.Array): normalization factor(s)
    '''

    # --- Definitions --------------------------------------------------------

    Ga = self.Ga
    Gb = self.Gb

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
        
    # --- Attributes ---------------------------------------------------------

    # --- Node attributes

    # Base
    N = np.ones((nA,nB))
    
    for k, attr in enumerate(Ga.vrtx_attr):

      bttr = Gb.vrtx_attr[k]

      if  attr['precision'] is None or attr['precision']:

        # Build contraint attribute
        A = np.tile(attr['values'], (nB,1)).transpose()
        B = np.tile(bttr['values'], (nA,1))
        N *= A==B
    
    # Remapping in [-1, 1]
    N = N*2 - 1

    # --- Edge attributes
    
    E = np.ones((mA,mB))

    # --- Computation --------------------------------------------------------

    if not mA or not mB:

      self.X = N
      self.Y = E

    else:

      # --- Initialization
      
      if not nIter:
        self.X = np.ones((nA, nB))  

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

        if Ga.directed:
          self.X = Ga.S @ self.Y @ Gb.S.T + Ga.T @ self.Y @ Gb.T.T
          self.Y = Ga.S.T @ self.X @ Gb.S + Ga.T.T @ self.X @ Gb.T

        else:
          self.X = Ga.R @ self.Y @ Gb.R.T
          self.Y = Ga.R.T @ self.X @ Gb.R

        # --- Normalization 

        if normalization is None:
          '''
          If normalization is not explicitely specified, the default normalization is used.
          '''
          self.X /= np.mean(self.X)
          self.Y /= np.mean(self.Y)

        else:
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
        print('Computing time', (time.perf_counter_ns()-t0)*1e-6, 'ms')

      # --- Post-processing
          
      self.X = self.X * N

  def compute_scores_GASM(self, GPU=True, **kwargs):
    ''' 
    Score computation with GASM (CPU and GPU)
    
    Parameters:
      'GPU' (bool): Run computations on CPU or GPU (default: True)
      'nIter' (int): Number of iterations
      'normalization' (float or np.Array): normalization factor(s)
      'eta' (float): Noise level (default 1e-10)
    '''

    # === Definitions ======================================================

    Ga = self.Ga
    Gb = self.Gb

    # --- Complements

    if 'complement' in kwargs:
      complement  = kwargs['complement']
      
    else:
      if Ga.directed:
        complement = Ga.nE + Gb.nE > (Ga.nV**2 + Gb.nV**2)/2
      else:
        complement = Ga.nE + Gb.nE > (Ga.nV*(Ga.nV+1) + Gb.nV*(Gb.nV+1))/4

    if complement:
      Ga = self.Ga.complement()
      Gb = self.Gb.complement()

    # --- Graph properties

    # Number of vertices
    nA = Ga.nV
    nB = Gb.nV

    # Number of edges    
    mA = Ga.nE
    mB = Gb.nE

    # --- Algorithms parameters

    self.info['GPU'] = GPU

    # Number of iterations
    nIter = kwargs['nIter'] if 'nIter' in kwargs else max(min(Ga.d, Gb.d), 1)
    self.info['nIter'] = nIter

    # Normalization
    if 'normalization' in kwargs:
      normalization = kwargs['normalization']
    else:
      normalization = 4*mA*mB/nA/nB if nA and nB else 1
      # normalization = 4*mA*mB/nA/nB + 1 if nA and nB else 1

    # Noise
    eta = kwargs['eta'] if 'eta' in kwargs else 1e-10
            
    # === Attributes =======================================================

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

        if attr['precision'] is None or attr['precision']:

          # --- Measurable attributes

          # Edge weights differences
          W = np.subtract.outer(wA, wB)

          if attr['precision'] is None:
            rho_a2 = np.var(W)
          else:
            rho_a2 = attr['precision']**2

          if rho_a2>0:
            N *= np.exp(-W**2/2/rho_a2)

        else:

          # --- Categorical attributes

          N *= np.equal.outer(wA, wB)

      # --- Edge attributes

      # Base
      E = np.ones((self.Ga.nE, self.Gb.nE))

      if self.Ga.nE and self.Gb.nE:

        for k, attr in enumerate(self.Ga.edge_attr):

          wA = attr['values']
          wB = self.Gb.edge_attr[k]['values']

          if attr['precision'] is None or attr['precision']:

            # --- Measurable attributes

            # Edge weights differences
            W = np.subtract.outer(wA, wB)

            if attr['precision'] is None:
              rho_a2 = np.var(W)
            else:
              rho_a2 = attr['precision']**2

            if rho_a2>0:
              E *= np.exp(-W**2/2/rho_a2)

          else:
            # --- Categorical attributes

            E *= np.equal.outer(wA, wB)

    # Random initial fluctuations
    H = np.random.rand(nA, nB)*eta

    # === Checks ===========================================================

    if not mA or not mB:
      
      self.X = N
      self.Y = E
      return

    # === Computation ======================================================
    
    if self.verbose:
      t0 = time.perf_counter_ns()

    if GPU:

      directed = Ga.directed

      blockDim = (16, 16)
      gridDim_X2Y = ((Ga.nV+(blockDim[0]-1))//blockDim[0], 
                     (Gb.nV+(blockDim[1]-1))//blockDim[1])
      gridDim_Y2X = ((Ga.nE+(blockDim[0]-1))//blockDim[0], 
                     (Gb.nE+(blockDim[1]-1))//blockDim[1])

      # --- CUDA Arrays ----------------------------------------------------
      
      # Scores
      d_X = cuda.to_device((N+H).astype(np.float32))
      d_Y = cuda.to_device(E.astype(np.float32))

      # --- Graph structure

      A_sn, A_src, A_tgt = Ga.to_CUDA_arrays()
      B_sn, B_src, B_tgt = Gb.to_CUDA_arrays()

      d_A_sn = cuda.to_device(A_sn)
      d_A_src = cuda.to_device(A_src)
      d_A_tgt = cuda.to_device(A_tgt)
      d_A_edges = cuda.to_device(Ga.edges.astype(np.int64))

      d_B_sn = cuda.to_device(B_sn)
      d_B_src = cuda.to_device(B_src)
      d_B_tgt = cuda.to_device(B_tgt)
      d_B_edges = cuda.to_device(Gb.edges.astype(np.int64))

      # --- Initial step

      Y2X[gridDim_Y2X, blockDim](d_X, d_Y, 
                                 d_A_sn, d_A_src, d_A_tgt,
                                 d_B_sn, d_B_src, d_B_tgt, 
                                 directed, 1, True)
       
      # --- Iterations

      for i in range(nIter):

        X2Y[gridDim_X2Y, blockDim](d_X, d_Y, 
                                   d_A_edges, d_B_edges,
                                   directed)

        Y2X[gridDim_Y2X, blockDim](d_X, d_Y, 
                                 d_A_sn, d_A_src, d_A_tgt,
                                 d_B_sn, d_B_src, d_B_tgt, 
                                 directed, normalization, False)

      # --- Get back scores to the host

      self.X = d_X.copy_to_host()

    else:

      # --- Initialization -------------------------------------------------
      
      # Define X0
      if Ga.directed:
        self.X = (self.Ga.S @ E @ self.Gb.S.T + self.Ga.T @ E @ self.Gb.T.T) * (N+H)
      else:
        self.X = (self.Ga.R @ E @ self.Gb.R.T) * (N+H)

      self.Y = np.ones((mA, mB))

      # --- Iterations

      for i in range(nIter):

        if Ga.directed:

          self.Y = Ga.S.T @ self.X @ Gb.S + Ga.T.T @ self.X @ Gb.T
          self.X = (Ga.S @ self.Y @ Gb.S.T + Ga.T @ self.Y @ Gb.T.T)

        else:

          self.Y = Ga.R.T @ self.X @ Gb.R
          self.X = Ga.R @ self.Y @ Gb.R.T

        # --- Normalization 
              
        if normalization is not None:
          self.X /= normalization

      # --- Information
          
      if 'info_avgScores' in kwargs:

        # Initialization
        if 'avgX' not in self.info:
          self.info['avgX'] = []

        # Update
        self.info['avgX'].append(np.mean(self.X))

    # --- Timing
        
    if self.verbose:
      print('Computing time', (time.perf_counter_ns()-t0)*1e-6, 'ms')

    # --- Post-processing
    
    # Isolated vertices
    I = self.X==0
    self.X[I] = N[I]

############################################################################
# ######################################################################## #
# #                                                                      # #
# #                              CUDA                                    # #
# #                                                                      # #
# ######################################################################## #
############################################################################

# --------------------------------------------------------------------------
#   The CUDA kernels
# --------------------------------------------------------------------------

@cuda.jit(cache=True)
def X2Y(X, Y, A_edges, B_edges, directed):

  i, j = cuda.grid(2)
  if i < Y.shape[0] and j < Y.shape[1]:

    if directed:

      Y[i,j] = X[A_edges[i,0], B_edges[j,0]] + X[A_edges[i,1], B_edges[j,1]]

    else:

      if A_edges[i,0]==A_edges[i,1]:

        if B_edges[j,0]==B_edges[j,1]:

          Y[i,j] = X[A_edges[i,0], B_edges[j,0]]

        else:

          Y[i,j] = X[A_edges[i,0], B_edges[j,0]] + X[A_edges[i,0], B_edges[j,1]]

      else:

        if B_edges[j,0]==B_edges[j,1]:

          Y[i,j] = X[A_edges[i,0], B_edges[j,0]] + X[A_edges[i,1], B_edges[j,0]]

        else:

          Y[i,j] = X[A_edges[i,0], B_edges[j,0]] + X[A_edges[i,1], B_edges[j,0]] + \
                   X[A_edges[i,0], B_edges[j,1]] + X[A_edges[i,1], B_edges[j,1]]
    
@cuda.jit(cache=True)
def Y2X(X, Y, A_sn, A_src, A_tgt, B_sn, B_src, B_tgt, directed, normalization, initialization):

  u, v = cuda.grid(2)
  if u < X.shape[0] and v < X.shape[1]:

    x = 0

    if directed:

      # Sources
      for i in range(A_sn[u,0], A_sn[u,0]+A_sn[u,1]):
        for j in range(B_sn[v,0], B_sn[v,0]+B_sn[v,1]):
          x += Y[A_src[i], B_src[j]]

      # Targets
      for i in range(A_sn[u,2], A_sn[u,2]+A_sn[u,3]):
        for j in range(B_sn[v,2], B_sn[v,2]+B_sn[v,3]):
          x += Y[A_tgt[i], B_tgt[j]]

    else:

      for i in range(A_sn[u,0], A_sn[u,0]+A_sn[u,1]):
        for j in range(B_sn[v,0], B_sn[v,0]+B_sn[v,1]):
          x += Y[A_tgt[i], B_tgt[j]]

    if normalization!=1:
      x /= normalization

    if initialization:
      X[u, v] *= x
    else:
      X[u, v] = x