import os

import project
from Network import *
from  Comparison import *

os.system('clear')

# --- Parameters -----------------------------------------------------------


# --------------------------------------------------------------------------

# Net = Network(nx=nx.random_lobster(10, 0.5, 0.5))

# Net = Network(nx=nx.duplication_divergence_graph(40, 0.2))

Net = star_branched(10,3)

print(Net)

Net.display()

