import os
import numpy as np

import project
import paprint as pa

os.system('clear')

w = 50
h = 50

M = np.random.rand(h,w)
# M = np.random.randint(-1100,100,(5,5))

# pa.line('Test', 1)

pa.matrix(M, highlight=M>0.5, title='This is a title')
