import time
import numpy as np
import numpy.linalg as la
from mpi4py import MPI

one_const = np.array([1], dtype=np.ulonglong)[0]

def chain_multiply_buffers(bufs):
    '''
    Multiplies elementwise all of the buffers in the provided
    list together.

    Assumption: the input array bufs must be at least length 1. 
    '''
    prod = np.ones_like(bufs[0], dtype=np.double) 

    for i in range(0, len(bufs)):
        prod = np.multiply(prod, bufs[i])

    return prod

def cl(n):
    return np.array([n], dtype=np.ulonglong)[0]

def round_to_nearest(n, m):
    return (n + m - 1) // m * m

def round_to_nearest_np_arr(n, m):
    n = np.array([n], dtype=np.ulonglong)[0]
    m = np.array([m], dtype=np.ulonglong)[0]

    return (n + m) // m * m

def compute_residual(ground_truth, current):
  return np.linalg.norm(ground_truth - current)

def get_norm_distributed(buf, world):
    val = la.norm(buf) ** 2
    result = np.zeros(1)
    world.Allreduce([val, MPI.DOUBLE], [result, MPI.DOUBLE]) 
    return np.sqrt(result)

def get_sum_distributed(buf, world):
    val = np.sum(buf)
    result = np.zeros(1)
    world.Allreduce([val, MPI.DOUBLE], [result, MPI.DOUBLE]) 
    return result 

def start_clock():
    return time.time()

def stop_clock_and_add(t0, dict, key):
    t1 = time.time()
    dict[key] += t1 - t0 