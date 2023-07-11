import os
import Network

os.system('clear')

Net = Network.Random(10, 0.5)

Sub = Net.subnet(5)

print(Net, Sub)