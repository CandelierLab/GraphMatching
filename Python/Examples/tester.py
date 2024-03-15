import project
from Examples.Example import *

algo = 'GASM'

Ex = Example(5.0)

# --- Matching

C = Comparison(Ex.Ga, Ex.Gb, algorithm=algo)

# C.compute_scores()

M = C.get_matching(force_perfect=False)

# --- Output

pa.line(os.path.basename(__file__))
print()

pa.matrix(C.X, title='Node scores')
# pa.matrix(C.Y)

print(M)