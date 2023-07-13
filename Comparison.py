import time
import pprint
import numpy as np
from scipy.optimize import linear_sum_assignment

# === Comparison ===========================================================
def compare(NetA, NetB, weight_constraint=True, nIter=100):
  '''
  Comparison of two networks.

  The algorithm is identical to [1] but with the addition of a constraint
  of edge weight similarity. Set weight_constraint=False to recover the 
  original algorithm.

  [1] L.A. Zager and G.C. Verghese, "Graph similarity scoring and matching",
      Applied Mathematics Letters 21 (2008) 86–94, doi: 10.1016/j.aml.2007.01.006
  '''
  
  # Define edges
  NetA.adj2edges()
  NetB.adj2edges()

  # --- Definitions

  # Number of nodes
  nA = NetA.nNd
  nB = NetB.nNd

  # Number of edges
  mA = NetA.nEd
  mB = NetB.nEd

  # --- Source-edge and terminus-edge matrices

  # Net A
  As = np.zeros((nA, mA))
  At = np.zeros((nA, mA))
  for k, e in enumerate(NetA.edge):
    As[e['i'], k] = 1
    At[e['j'], k] = 1

  # Net B
  Bs = np.zeros((nB, mB))
  Bt = np.zeros((nB, mB))
  for k, e in enumerate(NetB.edge):
    Bs[e['i'], k] = 1
    Bt[e['j'], k] = 1

  # --- Weight constraint

  if weight_constraint:

    # Edge weights
    W = np.zeros((mA, mB))
    for i, a in enumerate(NetA.edge):
      for j, b in enumerate(NetB.edge):
        W[i,j] = a['w'] - b['w']

    sigma2 = np.var(W)
    if sigma2>0:
      W = np.exp(-W**2/2/sigma2)
      yc = W.reshape(mA*mB)
    else:
      yc = np.ones(mA*mB)

  else:
    yc = np.ones(mA*mB)

  # --- Initialization

  x = np.ones(nA*nB)
  y = np.ones(mA*mB)

  # Structure matrix
  G = np.kron(As.T, Bs.T) + np.kron(At.T, Bt.T)

  for i in range(nIter):

    y_ = (G @ x) * yc
    x_ = G.T @ y

    # Normalization
    x = x_/np.sqrt(np.sum(x_**2))
    y = y_/np.sqrt(np.sum(y_**2))

  # Similarity matrices
  S_nodes = x.reshape((nA, nB))
  S_edges = y.reshape((mA, mB))

  return(S_nodes, S_edges)

# === Matching =============================================================
def matching(NetA, NetB, threshold=None, **kwargs):

  # Get similarity measures
  start = time.time()
  Sim = compare(NetA, NetB, **kwargs)[0]
  print('Scoring:', time.time()-start)

  # Threshold
  if threshold is not None:
    Sim[Sim<threshold] = -np.inf

  # Hungarian algorithm (Jonker-Volgenant)
  start = time.time()
  I, J = linear_sum_assignment(Sim, True)
  print('Matching:', time.time()-start)


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
