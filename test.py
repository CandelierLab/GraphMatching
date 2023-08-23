import os
from Network import *
from  Comparison import *
import timeit

os.system('clear')

Net = Network(10)
Net.set_rand_edges('ER', 0.1)

# print(Net)
Net.print()