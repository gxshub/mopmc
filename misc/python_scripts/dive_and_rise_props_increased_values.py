import math
import sys
import random

# Check the number of command-line arguments
num_args = len(sys.argv)

fname = sys.argv[1]        # The first argument is the filename
n_targets = int(sys.argv[2])    # Number of targets
inc_value = float(sys.argv[3])  # Increment of lower-bound values

y_max = 99  # must be the same as defined in the model file
inc_index = math.floor(y_max / (n_targets - 1))
with open(fname+'_'+str(n_targets)+'.props', 'w') as file:
    file.write("multi(\n")
    for i in range(n_targets-1):
        j = i * inc_index
        k = (i+1) * inc_value
        file.write(f'R{{\"target_{j}\"}}>={k} [ C ],\n')
    j = (n_targets - 1) * inc_index
    k = n_targets * inc_value
    file.write(f'R{{\"target_{j}\"}}>={k} [ C ]\n')
    file.write(")")





