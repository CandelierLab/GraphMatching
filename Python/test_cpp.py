from ctypes import CDLL, POINTER
from ctypes import c_size_t, c_double
import numpy as np

# load the library
mylib = CDLL("C++/gasp.so")

# C-type corresponding to numpy array 
ND_POINTER = np.ctypeslib.ndpointer(dtype=np.float64, 
                                      ndim=2,
                                      flags="C")

# define prototypes
mylib.print_matrix.argtypes = [ND_POINTER, c_size_t]
mylib.print_matrix.restype = None

mylib.timestwo.argtypes = [ND_POINTER, c_size_t]
mylib.timestwo.restype = ND_POINTER

X = np.ones((5,5))

Y = mylib.timestwo(X, *X.shape)

print(X)
print(Y)