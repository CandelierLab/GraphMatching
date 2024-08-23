from Example import Example
from Comparison import *

algo = 'GASM'

Ex = Example(2.0)

# Gb, gt = Ex.Ga.shuffle()

# --- Matching

C = Comparison(Ex.Ga, Ex.Gb)
M = C.get_matching(algorithm=algo, normalization=1, nIter=2)
M.compute_accuracy(Ex.gt)

# --- Output

pa.line(os.path.basename(__file__))
print()

pa.matrix(C.X, title='Node scores')
# pa.matrix(C.Y)

print(M)
