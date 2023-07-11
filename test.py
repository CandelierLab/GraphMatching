import os
import Network
from  Comparison import *

os.system('clear')

Net = Network.Random(10, 0.5)

Sub = Net.subnet(5)

S_nodes, S_edges = compare(Net, Sub)

print(S_nodes, S_edges)