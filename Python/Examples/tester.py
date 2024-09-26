from Example import Example
from Comparison import *

algo = 'GASM'

Ex = Example(2.2)

# Gb, gt = Ex.Ga.shuffle()

# --- Matching

C = Comparison(Ex.Ga, Ex.Gb)
M = C.get_matching(algorithm=algo, normalization=1, nIter=2)

M.idxB = [0, 2, 4, 3, 1]


M.compute_accuracy(Ex.gt)
M.compute_structural_quality()

# --- Output

pa.line(os.path.basename(__file__))
print()

pa.matrix(C.X, title='Node scores')
# pa.matrix(C.Y)

print(M)
