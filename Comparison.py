import time
import pprint
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import time

# === Comparison ===========================================================

def scores(NetA, NetB, weight_constraint=False, nIter=100):
  '''
  Comparison of two networks.

  The algorithm is identical to [1] but with the addition of a constraint
  of edge weight similarity. Set weight_constraint=False to recover the 
  original algorithm.

  [1] L.A. Zager and G.C. Verghese, "Graph similarity scoring and matching",
      Applied Mathematics Letters 21 (2008) 86–94, doi: 10.1016/j.aml.2007.01.006
  '''
  
  start = time.time()
  toc = lambda : print((time.time() - start)*1000)

  # --- Definitions

  # Number of nodes
  nA = NetA.nNd
  nB = NetB.nNd

  # Number of edges
  mA = NetA.nEd
  mB = NetB.nEd

  # Weights
  wA = 0 #NetA.edge_attr[0]
  wB = 0 #NetB.edge_attr[0]

  # toc()

  # --- Weight constraint

  if weight_constraint:

    # Edge weights differences
    W = np.subtract.outer(wA, wB)

    # Slightly slower implementation:
    # W = Wa[:,np.newaxis] - Wb

    sigma2 = np.var(W)
    if sigma2>0:
      Yc = np.exp(-W**2/2/sigma2)
    else:
      Yc = np.ones((mA,mB))

  else:
    Yc = np.ones((mA,mB))

  # toc()

  # --- Computation

  # print(nA, nB, mA, mB)

  X = np.ones((nA,nB))
  Y = np.ones((mA,mB))

  for i in range(nIter):

    Y_ = (NetA.As.T @ X @ NetB.As + NetA.At.T @ X @ NetB.At) * Yc
    X_ = NetA.As @ Y @ NetB.As.T + NetA.At @ Y @ NetB.At.T

    # Normalization
    X = X_/np.sqrt(np.sum(X_**2))
    Y = Y_/np.sqrt(np.sum(X_**2))

  # toc()

  return(X, Y)

# === Matching =============================================================
def matching(NetA, NetB, threshold=None, verbose=False, **kwargs):

  # Get similarity measures
  if verbose:
    start = time.time()

  Sim = scores(NetA, NetB, **kwargs)[0]

  if verbose:
    print('Scoring: {:.02f} ms'.format((time.time()-start)*1000), end=' - ')

  # Threshold
  if threshold is not None:
    Sim[Sim<threshold] = -np.inf

  # Hungarian algorithm (Jonker-Volgenant)
  if verbose:
    start = time.time()

  I, J = linear_sum_assignment(Sim, True)

  if verbose:
    print('Matching: {:.02f} ms'.format((time.time()-start)*1000))

  # Output
  M = [(I[k], J[k]) for k in range(len(I))]

  return M
  # return MatchNet(NetA, NetB, M)

# === MatchNet class ======================================================

class MatchNet():

  def __init__(self, NetA, NetB, M):

    #  --- Nets

    self.NetA = NetA
    self.NetB = NetB

    # --- Nodes and edges

    # Matched nodes
    self.mn = np.array(M)

    # Unmatched nodes
    self.unA = np.array([x for x in range(self.NetA.nNd) if x not in self.mn[:,0]])
    self.unB = np.array([x for x in range(self.NetB.nNd) if x not in self.mn[:,1]])

    # Matched edges
    me = []
    eB = np.array([[e['i'], e['j']] for e in self.NetB.edge])
    for u, e in enumerate(self.NetA.edge):
      i = self.mn[self.mn[:,0]==e['i'], 1]
      j = self.mn[self.mn[:,0]==e['j'], 1]
      if i.size and j.size:
        w = np.where((eB == (i[0], j[0])).all(axis=1))[0]
        if w.size: me.append((u, w[0]))
        
    self.me = np.array(me)

    # Unmatched edges
    self.ueA = np.array([x for x in range(self.NetA.nEd) if x not in self.me[:,0]])
    self.ueB = np.array([x for x in range(self.NetB.nEd) if x not in self.me[:,1]])

    # --- Ratios

    # Ratio of matched nodes
    self.rmn = self.mn.size/(self.mn.size + self.unA.size + self.unB.size)

    # Ratio of matched edges
    self.rme = self.me.size/(self.me.size + self.ueA.size + self.ueB.size)

    # --- Edge weight distances

    # Average matched edge weights distance
    wA = np.array([self.NetA.edge[i]['w'] for i in self.me[:,0]])
    wB = np.array([self.NetB.edge[j]['w'] for j in self.me[:,1]])
    # self.amewd = np.mean((np.abs(wA-wB)))
    self.amewd = np.mean((wA-wB)**2)

    # Average unmatched edge weights
    wA = np.array([self.NetA.edge[i]['w'] for i in self.unA])
    wB = np.array([self.NetB.edge[j]['w'] for j in self.unB])
    if wA.size or wB.size:
      self.auew = np.mean(np.abs(np.concatenate((wA, wB))))
    else:
      self.auew = None

  def print(self):

    pp = pprint.PrettyPrinter(depth=4)
    pp.pprint(self.__dict__)

    print('Matched node proportion: {:.2f}'.format(100*self.rmn))
    print('Matched egde proportion: {:.2f}'.format(100*self.rme))

    print('Average matched edge weight distance: {:.02f}'.format(self.amewd))
    if self.auew is not None:
      print('Average unmatched edge weight: {:.02f}'.format(self.auew))
