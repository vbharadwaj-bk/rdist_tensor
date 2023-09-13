PYTHONMALLOC=malloc valgrind \
        --suppressions=/global/homes/v/vbharadw/valgrind-python.supp --gen-suppressions=all \
        python -E test_modules.py


