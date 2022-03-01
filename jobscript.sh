
#mpirun -np 1 python3 synth_cubic.py -d 3 -s 27 -g 5 -t 2 -p 0.1 -iter 5

mpirun -np 1 python3 synth_cubic.py -d 3 -s 28 -g 5 -t 2 -p 0.1 -iter 5
mpirun -np 8 python3 synth_cubic.py -d 3 -s 28 -g 5 -t 2 -p 0.1 -iter 5
mpirun -np 27 python3 synth_cubic.py -d 3 -s 28 -g 5 -t 2 -p 0.1 -iter 5

#mpirun -np 8 python3 synth_cubic.py -d 3 -s 27 -g 5 -t 2 -p 0.1 -iter 5