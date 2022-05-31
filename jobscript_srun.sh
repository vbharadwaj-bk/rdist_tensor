#for proc_count in 64 216 512 1000
#do
#    srun -n $proc_count python3 synth_cubic.py -d 3 -s 1200 -p 0.1 -g 30 -t 30 -iter 50 -o data/sketchp1_strong.out
#    srun -n $proc_count python3 synth_cubic.py -d 3 -s 1200 -p 0.3 -g 30 -t 30 -iter 50 -o data/sketchp3_strong.out
#    srun -n $proc_count python3 synth_cubic.py -d 3 -s 1200 -p 0.5 -g 30 -t 30 -iter 50 -o data/sketchp5_strong.out
#    srun -n $proc_count python3 synth_cubic.py -d 3 -s 1200 -p 0.7 -g 30 -t 30 -iter 50 -o data/sketchp7_strong.out
#    srun -n $proc_count python3 synth_cubic.py -d 3 -s 1200 -g 30 -t 30 -iter 50 -o data/nosketch_strong.out
#done


#srun -n 64 python3 synth_cubic.py -d 3 -s 1200 -g 30 -t 30 -p 0.01 -iter 50 -o data/test.out
#srun -n 64 python3 synth_cubic.py -d 3 -s 1200 -g 30 -t 30 -iter 50 -o data/test.out


#for proc_count in 27 64 125 216
#do
#    srun -n $proc_count python3 synth_cubic.py -d 3 -s 1000 -g 30 -t 30 -iter 50 -o data/nosketch_strong.out
    #srun -n $proc_count python3 synth_cubic.py -d 3 -s 1000 -p 0.3 -g 30 -t 30 -iter 50 -o data/sketchp1_strong.out
    #srun -n $proc_count python3 synth_cubic.py -d 3 -s 1200 -p 0.3 -g 30 -t 30 -iter 50 -o data/sketchp3_strong.out
    #srun -n $proc_count python3 synth_cubic.py -d 3 -s 1200 -p 0.5 -g 30 -t 30 -iter 50 -o data/sketchp5_strong.out
    #srun -n $proc_count python3 synth_cubic.py -d 3 -s 1200 -p 0.7 -g 30 -t 30 -iter 50 -o data/sketchp7_strong.out
#done


srun -N 16 -n 512 python decompose_sparse.py -t 30 -iter 50 -o data/nell_test.out
