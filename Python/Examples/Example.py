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

        

