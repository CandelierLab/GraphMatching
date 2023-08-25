import os
from Network import *
from  Comparison import *
import timeit

os.system('clear')

Net = Network(20)
Net.set_rand_edges('ER', 0.1)
# Net.add_edge_attr('rand')

# Set, Idx = Net.subnet(5)
Det = Net.degrade('struct', p=0.5)

# print(Net)
Net.print()
Det.print()