import os

import project
from Network import *
from  Comparison import *

os.system('clear')

# --- Parameters -----------------------------------------------------------


# --------------------------------------------------------------------------

Net = Network(nx=nx.balanced_tree(2, 5))

Net.display()