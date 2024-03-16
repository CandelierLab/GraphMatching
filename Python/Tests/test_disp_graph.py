import os

import project
from Network import *
from  Comparison import *

os.system('clear')

# --- Parameters -----------------------------------------------------------


# --------------------------------------------------------------------------

Net = Network(nx=nx.circular_ladder_graph(3))

Net.display()