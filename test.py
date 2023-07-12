import os
import Network
from  Comparison import *

os.system('clear')

Net = Network.Random(10, 0.5)

Sub, Idx = Net.subnet(10)

S_nodes, S_edges = compare(Net, Sub)

print(S_nodes, Idx)