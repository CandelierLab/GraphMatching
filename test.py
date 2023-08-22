import os
from Network import *
from  Comparison import *
import timeit

os.system('clear')

Net = Network(5)
Net.set_rand_edges('ERG', 0.5)

print(Net)

# Net.print()