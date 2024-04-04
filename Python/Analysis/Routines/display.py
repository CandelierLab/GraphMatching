'''
                          Display routine

A routine to save the figures resulting of computations.
'''

import os
import project

os.system('clear')

# --- Parameters -----------------------------------------------------------

# Output base folder (in project/Figures)
basedir = 'Test'

# --- Functions ------------------------------------------------------------

def run(name, path):
  os.system(f'python Python/Analysis/Display/{path} -f "{name}"')

def check_folder(path):

  fdir = basedir + '/' + path

  # Create folders if not existing
  fpath = project.root + '/Figures/' + fdir
  if not os.path.exists(fpath):
    os.makedirs(fpath)

  return fdir

# --------------------------------------------------------------------------

# --- Self-matching

fdir = check_folder('Self-matching/')

run(fdir + 'gq_BT.png', 'Self-matching/gq_BT.py')
run(fdir + 'gq_CL.png', 'Self-matching/gq_CL.py')
run(fdir + 'gq_ER.png', 'Self-matching/gq_ER.py')
run(fdir + 'gq_SB.png', 'Self-matching/gq_SB.py')
