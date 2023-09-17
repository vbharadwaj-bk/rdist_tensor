. env.sh

python decompose.py -i uber \
                    --trank 25 \
                    -s 65536 \
                    -iter 40 \
                    -alg cp_arls_lev \
                    -dist tensor_stationary \
                    -r 1
#                    -o data/fit_progress_vs_time \
