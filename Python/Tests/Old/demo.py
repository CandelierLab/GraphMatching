import os

# import project
from Graph import *
from  Comparison import *

import paprint as pa

os.system('clear')

# === Input graphs =========================================================

'''
In this section you define the two graphs to match. It has to be instances 
of the Graph class, and you can import your graphs either from networkx or 
from the adjacency matrix.
'''

# ---- First graph

'''
Here I use a random graph with 1000 nodes and an average degree of 5 with
the dedicated networkx generation function.
'''
n = 150
d = n*0.75
Ga = Graph(nx=nx.gnp_random_graph(n, d/n, seed=np.random, directed=True))

# Add some attributes
'''
Please refer to the paper for the defintion of the error. An error of None
considers that the default error should be used, i.e. the standard deviation
over the values. For categorical attributes, usually an error of 0 is prefered.
'''
Ga.add_vrtx_attr({'measurable': True, 'error': 0.1, 'values': np.random.randn(Ga.nV), 'name': '1st attr'})
Ga.add_vrtx_attr({'measurable': False, 'error': 0, 'values': np.random.randn(Ga.nV), 'name': '2nd attr'})
Ga.add_edge_attr({'measurable': True, 'error': None, 'values': np.random.randn(Ga.nE), 'name': '3rd attr'})
Ga.add_edge_attr({'measurable': False, 'error': 0, 'values': np.random.randn(Ga.nE), 'name': 'last attr'})

# Display some infos
print(Ga)

# ---- Second graph

'''
Gb Can be defined just like Ga, but I have a convenient method to generate 
an isomorphoc graph.

NB: gt stands for "ground truth" and encapsulate the actual correspondence 
between the two graphs. You don't have access to this in practice, otherwise
you wouldn't need a matching algorithm.
'''

Gb, gt = Ga.shuffle()

# === Matching ===========================================================

C = Comparison(Ga, Gb, verbose=False)

M = C.get_matching(algorithm='GASM', GPU=True)

# You can comment this line if you don't have the ground truth, you won't have the accuracy
M.compute_accuracy(gt)

# Display matching infos
print(M)

'''
for further analyses, the matching is contained in M.idxA and M.idxB, which 
are the lists of corresponding nodes in Ga and Gb.
'''
