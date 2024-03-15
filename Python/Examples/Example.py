import os

import project 
from Network import *
from  Comparison import *
import paprint as pa

os.system('clear')

class Example():
  '''
  Test suite class
  '''

  def __init__(self, id=1.0):

    # Graphs
    self.Ga = None
    self.Gb = None

    # Identifier
    self.id = id

    # ######################################################################
    #                             EXAMPLE LIST
    # ######################################################################

    match self.id:

      case 1.0:

        nA = 4
        nB = 4

        self.Ga = Network(nA, directed=False)
        self.Ga.Adj = np.zeros((nA,nA), dtype=bool)
        self.Ga.Adj[0,1] = True
        self.Ga.Adj[1,2] = True
        self.Ga.Adj[2,3] = True

        self.Gb = Network(nB, directed=False)
        self.Gb.Adj = np.zeros((nB,nB), dtype=bool)
        self.Gb.Adj[0,1] = True
        self.Gb.Adj[1,2] = True
        self.Gb.Adj[2,3] = True

      case 1.1:

        nA = 4
        nB = 4

        self.Ga = Network(nA, directed=False)
        self.Ga.Adj = np.zeros((nA,nA), dtype=bool)
        self.Ga.Adj[0,1] = True
        self.Ga.Adj[1,2] = True
        self.Ga.Adj[2,3] = True
        self.Ga.add_node_attr({'measurable': False, 'values': [0, 1, 0, 0]})

        self.Gb = Network(nB, directed=False)
        self.Gb.Adj = np.zeros((nB,nB), dtype=bool)
        self.Gb.Adj[0,1] = True
        self.Gb.Adj[1,2] = True
        self.Gb.Adj[2,3] = True
        self.Gb.add_node_attr({'measurable': False, 'values': [0, 0, 0, 0]})

      case 1.2:

        nA = 4
        nB = 4

        self.Ga = Network(nA, directed=False)
        self.Ga.Adj = np.zeros((nA,nA), dtype=bool)
        self.Ga.Adj[0,1] = True
        self.Ga.Adj[1,2] = True
        self.Ga.Adj[2,3] = True
        self.Ga.add_node_attr({'measurable': False, 'values': [1, 0, 0, 0]})

        self.Gb = Network(nB, directed=False)
        self.Gb.Adj = np.zeros((nB,nB), dtype=bool)
        self.Gb.Adj[0,1] = True
        self.Gb.Adj[1,2] = True
        self.Gb.Adj[2,3] = True
        self.Gb.add_node_attr({'measurable': False, 'values': [0, 0, 0, 0]})

      case 1.3:

        nA = 4
        nB = 4

        self.Ga = Network(nA, directed=False)
        self.Ga.Adj = np.zeros((nA,nA), dtype=bool)
        self.Ga.Adj[0,1] = True
        self.Ga.Adj[1,2] = True
        self.Ga.Adj[2,3] = True
        self.Ga.add_node_attr({'measurable': False, 'values': [0, 1, 0, 0]})

        self.Gb = Network(nB, directed=False)
        self.Gb.Adj = np.zeros((nB,nB), dtype=bool)
        self.Gb.Adj[0,1] = True
        self.Gb.Adj[1,2] = True
        self.Gb.Adj[2,3] = True
        self.Gb.add_node_attr({'measurable': False, 'values': [0, 1, 0, 0]})
        
      case 2.0:

        nA = 5
        nB = 5

        self.Ga = Network(nA)
        self.Ga.Adj = np.zeros((nA,nA), dtype=bool)
        self.Ga.Adj[0,1] = True
        self.Ga.Adj[1,2] = True
        self.Ga.Adj[0,3] = True
        self.Ga.Adj[3,4] = True
        
        self.Gb = Network(nB)
        self.Gb.Adj = np.zeros((nB,nB), dtype=bool)
        self.Gb.Adj[0,1] = True
        self.Gb.Adj[1,2] = True
        self.Gb.Adj[0,3] = True
        self.Gb.Adj[3,4] = True
        
      case 2.1:

        nA = 5
        nB = 5

        self.Ga = Network(nA)
        self.Ga.Adj = np.zeros((nA,nA), dtype=bool)
        self.Ga.Adj[0,1] = True
        self.Ga.Adj[1,2] = True
        self.Ga.Adj[0,3] = True
        self.Ga.Adj[3,4] = True
        self.Ga.add_node_attr({'measurable': False, 'values': [0, 0, 0, 0, 0]})

        self.Gb = Network(nB)
        self.Gb.Adj = np.zeros((nB,nB), dtype=bool)
        self.Gb.Adj[0,1] = True
        self.Gb.Adj[1,2] = True
        self.Gb.Adj[0,3] = True
        self.Gb.Adj[3,4] = True
        self.Gb.add_node_attr({'measurable': False, 'values': [0, 1, 0, 0, 0]})

      case 2.2:

        nA = 5
        nB = 5

        self.Ga = Network(nA)
        self.Ga.Adj = np.zeros((nA,nA), dtype=bool)
        self.Ga.Adj[0,1] = True
        self.Ga.Adj[1,2] = True
        self.Ga.Adj[0,3] = True
        self.Ga.Adj[3,4] = True
        self.Ga.add_node_attr({'measurable': False, 'values': [0, 1, 0, 0, 0]})

        self.Gb = Network(nB)
        self.Gb.Adj = np.zeros((nB,nB), dtype=bool)
        self.Gb.Adj[0,1] = True
        self.Gb.Adj[1,2] = True
        self.Gb.Adj[0,3] = True
        self.Gb.Adj[3,4] = True
        self.Gb.add_node_attr({'measurable': False, 'values': [0, 1, 0, 0, 0]})

      case 2.3:

        nA = 5
        nB = 5

        self.Ga = Network(nA)
        self.Ga.Adj = np.zeros((nA,nA), dtype=bool)
        self.Ga.Adj[0,1] = True
        self.Ga.Adj[1,2] = True
        self.Ga.Adj[0,3] = True
        self.Ga.Adj[3,4] = True

        self.Gb = Network(nB)
        self.Gb.Adj = np.zeros((nB,nB), dtype=bool)
        self.Gb.Adj[0,1] = True
        self.Gb.Adj[1,2] = True
        self.Gb.Adj[4,3] = True
        self.Gb.Adj[3,0] = True

      case 3.0:

        nA = 5
        nB = 5

        self.Ga = Network(nA)
        self.Ga.Adj = np.zeros((nA,nA), dtype=bool)
        self.Ga.Adj[0,1] = True
        self.Ga.Adj[0,2] = True
        self.Ga.Adj[0,3] = True
        self.Ga.Adj[0,4] = True

        self.Gb = Network(nB)
        self.Gb.Adj = np.zeros((nB,nB), dtype=bool)
        self.Gb.Adj[0,1] = True
        self.Gb.Adj[0,2] = True
        self.Gb.Adj[0,3] = True
        self.Gb.Adj[0,4] = True

      case 3.1:

        nA = 5
        nB = 5

        self.Ga = Network(nA)
        self.Ga.Adj = np.zeros((nA,nA), dtype=bool)
        self.Ga.Adj[0,1] = True
        self.Ga.Adj[0,2] = True
        self.Ga.Adj[0,3] = True
        self.Ga.Adj[0,4] = True
        self.Ga.add_node_attr({'measurable': False, 'values': [0, 1, 0, 0, 0]})

        self.Gb = Network(nB)
        self.Gb.Adj = np.zeros((nB,nB), dtype=bool)
        self.Gb.Adj[0,1] = True
        self.Gb.Adj[0,2] = True
        self.Gb.Adj[0,3] = True
        self.Gb.Adj[0,4] = True
        self.Gb.add_node_attr({'measurable': False, 'values': [0, 1, 0, 0, 0]})
        
      case 3.2:

        nA = 5
        nB = 5

        self.Ga = Network(nA)
        self.Ga.Adj = np.zeros((nA,nA), dtype=bool)
        self.Ga.Adj[0,1] = True
        self.Ga.Adj[0,2] = True
        self.Ga.Adj[0,3] = True
        self.Ga.Adj[0,4] = True
        self.Ga.add_node_attr({'measurable': False, 'values': [0, 0, 0, 0, 0]})

        self.Gb = Network(nB)
        self.Gb.Adj = np.zeros((nB,nB), dtype=bool)
        self.Gb.Adj[0,1] = True
        self.Gb.Adj[0,2] = True
        self.Gb.Adj[0,3] = True
        self.Gb.Adj[0,4] = True
        self.Gb.add_node_attr({'measurable': False, 'values': [1, 0, 0, 0, 0]})

      case 4.0:

        nA = 3
        nB = 5

        self.Ga = Network(nA)
        self.Ga.Adj = np.zeros((nA,nA), dtype=bool)
        self.Ga.Adj[0,1] = True
        self.Ga.Adj[1,2] = True
        self.Ga.Adj[2,0] = True
        self.Ga.add_node_attr({'measurable': False, 'values': [1, 0, 0]})

        self.Gb = Network(nB)
        self.Gb.Adj = np.zeros((nB,nB), dtype=bool)
        self.Gb.Adj[0,1] = True
        self.Gb.Adj[1,2] = True
        self.Gb.Adj[2,3] = True
        self.Gb.Adj[3,4] = True
        self.Gb.Adj[4,0] = True
        self.Gb.add_node_attr({'measurable': False, 'values': [1, 0, 0, 0, 0]})

      case 5.0:

        nA = 5
        nB = 5

        self.Ga = Network(nA, directed=False)
        self.Ga.Adj = np.zeros((nA,nA), dtype=bool)
        self.Ga.Adj[0,2] = True
        self.Ga.Adj[0,3] = True
        self.Ga.Adj[0,4] = True
        self.Ga.Adj[1,2] = True
        self.Ga.Adj[1,3] = True
        self.Ga.Adj[1,4] = True
        self.Ga.Adj[2,4] = True
        self.Ga.Adj[3,4] = True

        self.Gb = Network(nB, directed=False)
        self.Gb.Adj = np.zeros((nB,nB), dtype=bool)
        self.Gb.Adj[0,2] = True
        self.Gb.Adj[0,3] = True
        self.Gb.Adj[0,4] = True
        self.Gb.Adj[1,2] = True
        self.Gb.Adj[1,3] = True
        self.Gb.Adj[1,4] = True
        self.Gb.Adj[2,4] = True
        self.Gb.Adj[3,4] = True

      case 5.1:

        nA = 5
        nB = 5

        self.Ga = Network(nA, directed=False)
        self.Ga.Adj = np.zeros((nA,nA), dtype=bool)
        self.Ga.Adj[0,2] = True
        self.Ga.Adj[0,3] = True
        self.Ga.Adj[0,4] = True
        self.Ga.Adj[1,2] = True
        self.Ga.Adj[1,3] = True
        self.Ga.Adj[1,4] = True
        self.Ga.Adj[2,4] = True
        self.Ga.Adj[3,4] = True
        self.Ga.add_node_attr({'measurable': False, 'values': [1, 0, 0, 0, 0]})

        self.Gb = Network(nB, directed=False)
        self.Gb.Adj = np.zeros((nB,nB), dtype=bool)
        self.Gb.Adj[0,2] = True
        self.Gb.Adj[0,3] = True
        self.Gb.Adj[0,4] = True
        self.Gb.Adj[1,2] = True
        self.Gb.Adj[1,3] = True
        self.Gb.Adj[1,4] = True
        self.Gb.Adj[2,4] = True
        self.Gb.Adj[3,4] = True
        self.Gb.add_node_attr({'measurable': False, 'values': [1, 0, 0, 0, 0]})

      case 5.2:

        nA = 5
        nB = 5

        self.Ga = Network(nA, directed=False)
        self.Ga.Adj = np.zeros((nA,nA), dtype=bool)
        self.Ga.Adj[0,2] = True
        self.Ga.Adj[0,3] = True
        self.Ga.Adj[0,4] = True
        self.Ga.Adj[1,2] = True
        self.Ga.Adj[1,3] = True
        self.Ga.Adj[1,4] = True
        self.Ga.Adj[2,3] = True
        self.Ga.Adj[2,4] = True
        self.Ga.Adj[3,4] = True

        self.Gb = Network(nB, directed=False)
        self.Gb.Adj = np.zeros((nB,nB), dtype=bool)
        self.Gb.Adj[0,2] = True
        self.Gb.Adj[0,3] = True
        self.Gb.Adj[0,4] = True
        self.Gb.Adj[1,2] = True
        self.Gb.Adj[1,3] = True
        self.Gb.Adj[1,4] = True
        self.Gb.Adj[2,3] = True
        self.Gb.Adj[2,4] = True
        self.Gb.Adj[3,4] = True

      case 6.0:

        nA = 4
        nB = 4

        self.Ga = Network(nA, directed=False)
        self.Ga.Adj = np.zeros((nA,nA), dtype=bool)
        self.Ga.Adj[0,1] = True
        self.Ga.Adj[1,2] = True
        self.Ga.Adj[2,3] = True
        self.Ga.Adj[3,0] = True

        self.Gb = Network(nB, directed=False)
        self.Gb.Adj = np.zeros((nB,nB), dtype=bool)
        self.Gb.Adj[0,1] = True
        self.Gb.Adj[1,2] = True
        self.Gb.Adj[2,3] = True
        self.Gb.Adj[3,0] = True

      case 7.0:

        nA = 7
        nB = 7

        self.Ga = Network(nA)
        self.Ga.Adj = np.zeros((nA,nA), dtype=bool)
        self.Ga.Adj[0,1] = True
        self.Ga.Adj[0,2] = True
        self.Ga.Adj[0,3] = True
        self.Ga.Adj[1,4] = True
        self.Ga.Adj[2,5] = True
        self.Ga.Adj[3,6] = True

        self.Gb = Network(nB)
        self.Gb.Adj = np.zeros((nB,nB), dtype=bool)
        self.Gb.Adj[0,1] = True
        self.Gb.Adj[0,2] = True
        self.Gb.Adj[0,3] = True
        self.Gb.Adj[1,4] = True
        self.Gb.Adj[2,5] = True
        self.Gb.Adj[3,6] = True

    # Prepare networks
    self.Ga.prepare()
    self.Gb.prepare()

