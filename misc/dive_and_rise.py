import sys
import random

# Check the number of command-line arguments
num_args = len(sys.argv)

fname = sys.argv[1]                # The first argument is the filename
x_max = 199 #int(sys.argv[2])
x_target = 99 #int(sys.argv[3])
y_max = 99 #int(sys.argv[4])
p = 0.65 #float(sys.argv[5])
q = 0.20 #float(sys.argv[6])

with open(fname + '.nm', 'w') as file:
    file.write('mdp\n\n')

    # Set the constants
    file.write('//constants\n')
    file.write(f'const int x_max = {x_max};\n')
    file.write(f'const int x_target = {x_target};\n')
    file.write(f'const int y_max = {y_max};\n')
    file.write(f'const double p = {p};\n')
    file.write(f'const double q = {q};\n\n')

    file.write('module Dive_and_Rise\n\n')

    file.write('x : [0..x_max] init 0; //step\n')
    file.write('y : [0..y_max] init 0; //height\n')
    file.write('z : [0..1] init 0; //game on or not\n\n')
    file.write("[dive] x < x_max & z = 0 ->\n")
    file.write("\t p : (x'= x+1) & (y'= max(y-1, 0)) +\n")
    file.write("\t q : (x'= x+1) & (y'= y) +\n")
    file.write("\t 1-p-q : (x'= x+1) & (y'= min(y+1, y_max)) ;\n")
    file.write("[rise]  x < x_max & z = 0 ->\n")
    file.write("\t p : (x'= x+1) & (y'= min(y+1, y_max)) +\n")
    file.write("\t q : (x'= x+1) & (y'= y) +\n")
    file.write("\t 1-p-q : (x'= x+1) & (y'= max(y-1, 0)) ;\n")
    file.write("[end] x = x_max -> (x'= 0) & (y'= 0) & (z'= 1) ;\n")
    file.write("[none] z = 1 -> 1 : true ;\n\n")
    file.write("endmodule\n\n")

    for i in range(y_max+1):
        file.write(f"rewards \"target_{i}\"\n")
        file.write(f'\t (x=x_max)&(y={i})&(z=0): 1 ;\n')
        file.write("endrewards\n")



