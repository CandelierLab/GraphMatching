'''
                          Computation routine

A routine to call all the important computation programs.
'''

import os

# === Parameters ===========================================================

force = False

# ==========================================================================

# Run function
run = lambda name, path : os.system(f'tsp -L {name} python Python/Analysis/{path} ' + ('-F ' if force else '') + '> /dev/null')

os.system('clear')

# Set the number of jobs on the number of CPU threads
# os.system('tsp -S 28')
os.system('tsp -S 16')

# Self-matching
run('Self-Matching_BT', 'Computation/Self-matching/gq_BT.py')
# run('Self-Matching_CL', 'Computation/Self-matching/gq_CL.py')
# run('Self-Matching_ER', 'Computation/Self-matching/gq_ER.py')
# run('Self-Matching_SB', 'Computation/Self-matching/gq_SB.py')

# Displat tsp running list
os.system('tsp')