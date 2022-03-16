srun -n 1 python3 synth_cubic.py -d 3 -s 80 -g 20 -t 15 -p 0.1 -iter 50 -o data/strong_scaling.out

#srun -n 8 python3 synth_cubic.py -d 3 -s 80 -g 20 -t 3 -p 0.1 -iter 20

#srun -n 27 python3 synth_cubic.py -d 3 -s 80 -g 20 -t 3 -p 0.1 -iter 20
