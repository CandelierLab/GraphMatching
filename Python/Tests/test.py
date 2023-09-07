import os
import project

from Network import *
from  Comparison import *

os.system('clear')

nA = 3
nB = 4

nIter = 10
f = 1

NetA = Network(nA)
NetA.Adj = np.zeros((nA, nA))
NetA.Adj[1,0] = True
NetA.Adj[0,2] = True
NetA.nEd = 2
NetA.prepare()

NetB = Network(nB)
NetB.Adj = np.zeros((nB, nB))
NetB.Adj[1,0] = True
NetB.Adj[0,2] = True
NetB.Adj[0,3] = True
NetB.nEd = 3
NetB.prepare()

NetA.print()
NetB.print()

for iter in range(0, nIter):

  # Structure & attribute scores
  X, Y = scores(NetA, NetA, nIter=iter+1, normalization=f)

  print('---------------------------------------')
  print(X, X/np.mean(X))
  print('')
  print(Y, Y/np.mean(Y))
