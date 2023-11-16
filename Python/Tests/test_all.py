import os
import project

from Network import *
from Comparison import *

os.system('clear')

S = np.array([[2, 0, 3],[1, 2, 0],[0, 2, 2]])
I = [0, 1, 2]
# J = [0, 1, 2]
J = [2, 0, 1]

allsolutions(S, I, J)
