import os
import time
import numpy as np

import project
from Network import *
import paprint as pa

os.system('clear')

# --- Parameters -----------------------------------------------------------

np.random.seed(seed=0)

# --------------------------------------------------------------------------

# Bipartite matrix
B = np.random.rand(5,5)>0.5

pa.matrix(B)