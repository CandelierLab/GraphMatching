import os
import time
import pandas as pd

import project
from Graph import *
from Comparison import *
from QAPLIB import QAPLIB

import paprint as pa

os.system('clear')

# === Parameters ===========================================================

id = 0 #'lipa20a'

algo = 'GASM'

np.random.seed(0)

# --------------------------------------------------------------------------

fname = project.root + f'/Files/QAPLIB/sgq.csv'

# ==========================================================================

ref = time.time()

Q = QAPLIB()
B = []

for id in Q.l_inst:

  print(id, end='', flush=True)
  start = time.time()

  # Get instance
  I = Q.get(id)
  Ga, Gb, gt = Q.get_graphs(id)

  # === Solution

  s_sol = np.trace(I.A.T @ I.B[I.s, :][:, I.s])

  M = Matching(Ga, Gb)
  M.from_lists(list(range(I.n)), I.s)
  M.compute_accuracy(gt)
  g_sol = M.accuracy
  q_sol = M.structural_quality

  # === FAQ

  C = Comparison(Ga, Gb)
  M = C.get_matching(algorithm='FAQ')
  M.compute_accuracy(gt)

  s_FAQ = np.trace(I.A.T @ I.B[M.idxB, :][:, M.idxB])
  g_FAQ = M.accuracy
  q_FAQ = M.structural_quality
 
  # === 2opt

  C = Comparison(Ga, Gb)
  M = C.get_matching(algorithm='2opt')
  M.compute_accuracy(gt)

  s_2opt = np.trace(I.A.T @ I.B[M.idxB, :][:, M.idxB])
  g_2opt = M.accuracy
  q_2opt = M.structural_quality

  # === Zager

  C = Comparison(Ga, Gb)
  M = C.get_matching(algorithm='Zager')
  M.compute_accuracy(gt)

  s_Zager = np.trace(I.A.T @ I.B[M.idxB, :][:, M.idxB])
  g_Zager = M.accuracy
  q_Zager = M.structural_quality

  # === GASM

  C = Comparison(Ga, Gb)
  M = C.get_matching(algorithm='GASM')
  M.compute_accuracy(gt)

  s_GASM = np.trace(I.A.T @ I.B[M.idxB, :][:, M.idxB])
  g_GASM = M.accuracy
  q_GASM = M.structural_quality

  # === Update

  B.append([id, s_sol, s_FAQ, s_2opt, s_Zager, s_GASM, g_sol, g_FAQ, g_2opt, g_Zager, g_GASM, q_sol, q_FAQ, q_2opt, q_Zager, q_GASM])

  print('\t\t{:.02f} sec'.format((time.time() - start)))

df = pd.DataFrame(B, columns=['id', 's_sol', 's_FAQ', 's_2opt', 's_Zager', 's_GASM', 'g_sol', 'g_FAQ', 'g_2opt', 'g_Zager', 'g_GASM', 'q_sol', 'q_FAQ', 'q_2opt', 'q_Zager', 'q_GASM'])

print('Total time: {:.02f} sec'.format((time.time() - ref)))

# --- Save
    
print('Saving ...', end='')
ref = time.time()

df.to_csv(fname)

print('{:.02f} sec'.format((time.time() - ref)))
