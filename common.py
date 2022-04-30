import numpy as np
import numpy.linalg as la
from mpi4py import MPI

one_const = np.array([1], dtype=np.ulonglong)[0]

def round_to_nearest(n, m):
    return (n + m - 1) // m * m

def round_to_nearest_np_arr(n, m):
    return (n + m - one_const) // m * m

def compute_residual(ground_truth, current):
  return np.linalg.norm(ground_truth - current)

def get_norm_distributed(buf, world):
    val = la.norm(buf) ** 2
    result = np.zeros(1)
    world.Allreduce([val, MPI.DOUBLE], [result, MPI.DOUBLE]) 
    return np.sqrt(result)