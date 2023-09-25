import os
import numpy as np

import project
import paprint as pa

os.system('clear')

w = 5
h = 5

M = np.random.rand(h,w)
# M = np.random.randint(-1100,100,(5,5))

pa.matrix(M)
