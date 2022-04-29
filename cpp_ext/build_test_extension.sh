module load python
conda activate tensor_env

export CC=CC
export CXX=CC
python setup.py build_ext -i