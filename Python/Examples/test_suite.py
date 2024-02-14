import os

import project 
from Network import *
from  Comparison import *
import paprint as pa

os.system('clear')

class test_suite():
  '''
  Test suite class
  '''

  def __init__(self, seed=0):

    # Graphs
    self.Ga = None
    self.Gb = None

    # Initialize test
    self.index = seed
    self.set()
