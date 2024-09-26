import os

from Graph import *
import paprint as pa

os.system('clear')

class Example():
  '''
  Test suite class
  '''

  def __init__(self, id=1.0):

    # Identifier
    self.id = id

    # Graphs
    self.Ga = None
    self.Gb = None

    # Ground truth
    self.gt = None

    # ######################################################################
    #                             EXAMPLE LIST
    # ######################################################################

    match self.id:

      case 1.0:

        directed = False

        nA = 4
        nB = 4

        Adj_A = np.zeros((nA,nA), dtype=bool)
        Adj_A[0,1] = True
        Adj_A[1,2] = True
        Adj_A[2,3] = True

        Adj_B = np.zeros((nB,nB), dtype=bool)
        Adj_B[0,1] = True
        Adj_B[1,2] = True
        Adj_B[2,3] = True

      case 1.1:

        nA = 4
        nB = 4

        self.Ga = Graph(nA, directed=False)
        Adj_A = np.zeros((nA,nA), dtype=bool)
        Adj_A[0,1] = True
        Adj_A[1,2] = True
        Adj_A[2,3] = True
        self.Ga.add_node_attr({'measurable': False, 'values': [0, 1, 0, 0]})

        self.Gb = Graph(nB, directed=False)
        Adj_B = np.zeros((nB,nB), dtype=bool)
        Adj_B[0,1] = True
        Adj_B[1,2] = True
        Adj_B[2,3] = True
        self.Gb.add_node_attr({'measurable': False, 'values': [0, 0, 0, 0]})

      case 1.2:

        nA = 4
        nB = 4

        self.Ga = Graph(nA, directed=False)
        Adj_A = np.zeros((nA,nA), dtype=bool)
        Adj_A[0,1] = True
        Adj_A[1,2] = True
        Adj_A[2,3] = True
        self.Ga.add_node_attr({'measurable': False, 'values': [1, 0, 0, 0]})

        self.Gb = Graph(nB, directed=False)
        Adj_B = np.zeros((nB,nB), dtype=bool)
        Adj_B[0,1] = True
        Adj_B[1,2] = True
        Adj_B[2,3] = True
        self.Gb.add_node_attr({'measurable': False, 'values': [0, 0, 0, 0]})

      case 1.3:

        nA = 4
        nB = 4

        self.Ga = Graph(nA, directed=False)
        Adj_A = np.zeros((nA,nA), dtype=bool)
        Adj_A[0,1] = True
        Adj_A[1,2] = True
        Adj_A[2,3] = True
        self.Ga.add_node_attr({'measurable': False, 'values': [0, 1, 0, 0]})

        self.Gb = Graph(nB, directed=False)
        Adj_B = np.zeros((nB,nB), dtype=bool)
        Adj_B[0,1] = True
        Adj_B[1,2] = True
        Adj_B[2,3] = True
        self.Gb.add_node_attr({'measurable': False, 'values': [0, 1, 0, 0]})
        
      case 2.0:

        directed = True

        nA = 5
        nB = 5

        # Adj_A = np.zeros((nA,nA), dtype=bool)
        # Adj_A[0,1] = True
        # Adj_A[1,2] = True
        # Adj_A[0,3] = True
        # Adj_A[3,4] = True
        
        # Adj_B = np.zeros((nB,nB), dtype=bool)
        # Adj_B[0,1] = True
        # Adj_B[1,2] = True
        # Adj_B[0,3] = True
        # Adj_B[3,4] = True
        
        Adj_A = np.zeros((nA,nA), dtype=bool)
        Adj_A[0,1] = True
        Adj_A[0,2] = True
        Adj_A[1,3] = True
        Adj_A[2,4] = True
        
        Adj_B = np.zeros((nB,nB), dtype=bool)
        Adj_B[0,1] = True
        Adj_B[0,2] = True
        Adj_B[1,3] = True
        Adj_B[2,4] = True

      case 2.1:

        directed = True

        nA = 5
        nB = 5

        self.Ga = Graph(nA)
        Adj_A = np.zeros((nA,nA), dtype=bool)
        # Adj_A[0,1] = True
        # Adj_A[1,2] = True
        # Adj_A[0,3] = True
        # Adj_A[3,4] = True
        Adj_A[0,1] = True
        Adj_A[0,2] = True
        Adj_A[1,3] = True
        Adj_A[2,4] = True
        self.Ga.add_vrtx_attr({'measurable': False, 'error': None, 'values': [0, 0, 0, 0, 0]})

        self.Gb = Graph(nB)
        Adj_B = np.zeros((nB,nB), dtype=bool)
        # Adj_B[0,1] = True
        # Adj_B[1,2] = True
        # Adj_B[0,3] = True
        # Adj_B[3,4] = True
        Adj_B[0,1] = True
        Adj_B[0,2] = True
        Adj_B[1,3] = True
        Adj_B[2,4] = True
        self.Gb.add_vrtx_attr({'measurable': False, 'error': None, 'values': [0, 1, 0, 0, 0]})

      case 2.2:

        directed = True

        nA = 5
        nB = 5

        self.Ga = Graph(nA)
        Adj_A = np.zeros((nA,nA), dtype=bool)
        # Adj_A[0,1] = True
        # Adj_A[1,2] = True
        # Adj_A[0,3] = True
        # Adj_A[3,4] = True
        Adj_A[0,1] = True
        Adj_A[0,2] = True
        Adj_A[1,3] = True
        Adj_A[2,4] = True
        self.Ga.add_vrtx_attr({'measurable': False, 'error': None, 'values': [0, 1, 0, 0, 0]})

        self.Gb = Graph(nB)
        Adj_B = np.zeros((nB,nB), dtype=bool)
        # Adj_B[0,1] = True
        # Adj_B[1,2] = True
        # Adj_B[0,3] = True
        # Adj_B[3,4] = True
        Adj_B[0,1] = True
        Adj_B[0,2] = True
        Adj_B[1,3] = True
        Adj_B[2,4] = True
        self.Gb.add_vrtx_attr({'measurable': False, 'error': None, 'values': [0, 1, 0, 0, 0]})

      case 2.3:

        nA = 5
        nB = 5

        self.Ga = Graph(nA)
        Adj_A = np.zeros((nA,nA), dtype=bool)
        Adj_A[0,1] = True
        Adj_A[1,2] = True
        Adj_A[0,3] = True
        Adj_A[3,4] = True

        self.Gb = Graph(nB)
        Adj_B = np.zeros((nB,nB), dtype=bool)
        Adj_B[0,1] = True
        Adj_B[1,2] = True
        Adj_B[4,3] = True
        Adj_B[3,0] = True

      case 3.0:

        nA = 5
        nB = 5

        self.Ga = Graph(nA)
        Adj_A = np.zeros((nA,nA), dtype=bool)
        Adj_A[0,1] = True
        Adj_A[0,2] = True
        Adj_A[0,3] = True
        Adj_A[0,4] = True

        self.Gb = Graph(nB)
        Adj_B = np.zeros((nB,nB), dtype=bool)
        Adj_B[0,1] = True
        Adj_B[0,2] = True
        Adj_B[0,3] = True
        Adj_B[0,4] = True

      case 3.1:

        nA = 5
        nB = 5

        self.Ga = Graph(nA)
        Adj_A = np.zeros((nA,nA), dtype=bool)
        Adj_A[0,1] = True
        Adj_A[0,2] = True
        Adj_A[0,3] = True
        Adj_A[0,4] = True
        self.Ga.add_node_attr({'measurable': False, 'values': [0, 1, 0, 0, 0]})

        self.Gb = Graph(nB)
        Adj_B = np.zeros((nB,nB), dtype=bool)
        Adj_B[0,1] = True
        Adj_B[0,2] = True
        Adj_B[0,3] = True
        Adj_B[0,4] = True
        self.Gb.add_node_attr({'measurable': False, 'values': [0, 1, 0, 0, 0]})
        
      case 3.2:

        nA = 5
        nB = 5

        self.Ga = Graph(nA)
        Adj_A = np.zeros((nA,nA), dtype=bool)
        Adj_A[0,1] = True
        Adj_A[0,2] = True
        Adj_A[0,3] = True
        Adj_A[0,4] = True
        self.Ga.add_node_attr({'measurable': False, 'values': [0, 0, 0, 0, 0]})

        self.Gb = Graph(nB)
        Adj_B = np.zeros((nB,nB), dtype=bool)
        Adj_B[0,1] = True
        Adj_B[0,2] = True
        Adj_B[0,3] = True
        Adj_B[0,4] = True
        self.Gb.add_node_attr({'measurable': False, 'values': [1, 0, 0, 0, 0]})

      case 4.0:

        nA = 3
        nB = 5

        self.Ga = Graph(nA)
        Adj_A = np.zeros((nA,nA), dtype=bool)
        Adj_A[0,1] = True
        Adj_A[1,2] = True
        Adj_A[2,0] = True
        self.Ga.add_node_attr({'measurable': False, 'values': [1, 0, 0]})

        self.Gb = Graph(nB)
        Adj_B = np.zeros((nB,nB), dtype=bool)
        Adj_B[0,1] = True
        Adj_B[1,2] = True
        Adj_B[2,3] = True
        Adj_B[3,4] = True
        Adj_B[4,0] = True
        self.Gb.add_node_attr({'measurable': False, 'values': [1, 0, 0, 0, 0]})

      case 5.0:

        nA = 5
        nB = 5

        self.Ga = Graph(nA, directed=False)
        Adj_A = np.zeros((nA,nA), dtype=bool)
        Adj_A[0,2] = True
        Adj_A[0,3] = True
        Adj_A[0,4] = True
        Adj_A[1,2] = True
        Adj_A[1,3] = True
        Adj_A[1,4] = True
        Adj_A[2,4] = True
        Adj_A[3,4] = True

        self.Gb = Graph(nB, directed=False)
        Adj_B = np.zeros((nB,nB), dtype=bool)
        Adj_B[0,2] = True
        Adj_B[0,3] = True
        Adj_B[0,4] = True
        Adj_B[1,2] = True
        Adj_B[1,3] = True
        Adj_B[1,4] = True
        Adj_B[2,4] = True
        Adj_B[3,4] = True

      case 5.1:

        nA = 5
        nB = 5

        self.Ga = Graph(nA, directed=False)
        Adj_A = np.zeros((nA,nA), dtype=bool)
        Adj_A[0,2] = True
        Adj_A[0,3] = True
        Adj_A[0,4] = True
        Adj_A[1,2] = True
        Adj_A[1,3] = True
        Adj_A[1,4] = True
        Adj_A[2,4] = True
        Adj_A[3,4] = True
        self.Ga.add_node_attr({'measurable': False, 'values': [1, 0, 0, 0, 0]})

        self.Gb = Graph(nB, directed=False)
        Adj_B = np.zeros((nB,nB), dtype=bool)
        Adj_B[0,2] = True
        Adj_B[0,3] = True
        Adj_B[0,4] = True
        Adj_B[1,2] = True
        Adj_B[1,3] = True
        Adj_B[1,4] = True
        Adj_B[2,4] = True
        Adj_B[3,4] = True
        self.Gb.add_node_attr({'measurable': False, 'values': [1, 0, 0, 0, 0]})

      case 5.2:

        nA = 5
        nB = 5

        self.Ga = Graph(nA, directed=False)
        Adj_A = np.zeros((nA,nA), dtype=bool)
        Adj_A[0,2] = True
        Adj_A[0,3] = True
        Adj_A[0,4] = True
        Adj_A[1,2] = True
        Adj_A[1,3] = True
        Adj_A[1,4] = True
        Adj_A[2,3] = True
        Adj_A[2,4] = True
        Adj_A[3,4] = True

        self.Gb = Graph(nB, directed=False)
        Adj_B = np.zeros((nB,nB), dtype=bool)
        Adj_B[0,2] = True
        Adj_B[0,3] = True
        Adj_B[0,4] = True
        Adj_B[1,2] = True
        Adj_B[1,3] = True
        Adj_B[1,4] = True
        Adj_B[2,3] = True
        Adj_B[2,4] = True
        Adj_B[3,4] = True

      case 6.0:

        nA = 4
        nB = 4

        self.Ga = Graph(nA, directed=False)
        Adj_A = np.zeros((nA,nA), dtype=bool)
        Adj_A[0,1] = True
        Adj_A[1,2] = True
        Adj_A[2,3] = True
        Adj_A[3,0] = True

        self.Gb = Graph(nB, directed=False)
        Adj_B = np.zeros((nB,nB), dtype=bool)
        Adj_B[0,1] = True
        Adj_B[1,2] = True
        Adj_B[2,3] = True
        Adj_B[3,0] = True

      case 7.0:

        nA = 7
        nB = 7

        self.Ga = Graph(nA)
        Adj_A = np.zeros((nA,nA), dtype=bool)
        Adj_A[0,1] = True
        Adj_A[0,2] = True
        Adj_A[0,3] = True
        Adj_A[1,4] = True
        Adj_A[2,5] = True
        Adj_A[3,6] = True

        self.Gb = Graph(nB)
        Adj_B = np.zeros((nB,nB), dtype=bool)
        Adj_B[0,1] = True
        Adj_B[0,2] = True
        Adj_B[0,3] = True
        Adj_B[1,4] = True
        Adj_B[2,5] = True
        Adj_B[3,6] = True

    # Prepare Graphs
    self.Ga = Graph(nA, directed=directed, Adj=Adj_A)
    self.Gb = Graph(nB, directed=directed, Adj=Adj_B)

    if self.gt is None:
      self.gt = GroundTruth(self.Ga, self.Gb)
