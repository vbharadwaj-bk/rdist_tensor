# Distributed Randomized Algorithms for Sparse Tensor CP Decomposition

This repository contains code for the paper
[Distributed-Memory Randomized Algorithms for Sparse CP Decomposition](https://arxiv.org/abs/2210.05105), to appear at SPAA 2024. 


## What can I do with it?
You can test two randomized algorithms, CP-ARLS-LEV and STS-CP, that can decompose
billion-scale sparse tensors on a cluster of CPUs.

## Building our code 

### Step 0: Clone the repository
Clone the repo and `cd` into it.

```shell
[zsh]> git clone https://github.com/vbharadwaj-bk/fast_tensor_leverage.git
[zsh]> cd fast_tensor_leverage
```

### Step 1: Install Python packages
Install Python dependencies with the following command:
```shell
[zsh]> pip install -r requirements.txt
```
We rely on the Pybind11 and cppimport packages. We
use the HDF5 format to store sparse tensors, so
you need the h5py package if you want to perform
sparse tensor decomposition. 

### Step 2: Configure the compile and runtime environments 
Within the repository, run the following command:

```shell
[zsh]> python configure.py
```

