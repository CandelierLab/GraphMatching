'''
FlyWire

A class to magane the Flywire challenge
https://codex.flywire.ai/app/vnc_matching_challenge
'''

import pandas as pd
from alive_progress import alive_bar

from Graph import *
import project

class FlyWire:

  def __init__(self):

    # Load data
    self.load_data()

  def load_data(self):
    '''
    Loading the challenge's data
    '''

    # --- File paths

    fMale = project.root + '/Files/FlyWire/male_connectome_graph.csv'
    fFemale = project.root + '/Files/FlyWire/female_connectome_graph.csv'
    fBenchmark = project.root + '/Files/FlyWire/benchmark.csv'

    # --- Male graph

    self.Ga = None

    # Dataframe
    df = pd.read_csv(fMale)
    df.columns = ['i', 'j', 'w']

    # Vertices indices
    Anames = np.unique(np.concatenate((df.i.unique(), df.j.unique())))
    nV = len(Anames)
    Index = {name:i for i, name in enumerate(Anames)}
    
    # Adjacency matrix
    
    A = np.zeros((nV, nV), dtype=int)
    with alive_bar(len(df)) as bar:  # declare your expected total
      for index, row in df.iterrows():  
        A[Index[row['i']], Index[row['j']]] = int(row['w'])
        bar()
      
        if index>50000: break

    self.Ga = Graph(A.shape[0], directed=True, Adj=A>0)

    # Ga.add_edge_attr({'measurable': True,
    #                   'error':precision[0],
    #                   'values':[I.A[e[0], e[1]] for e in Ga.edges]})
