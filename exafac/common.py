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

def allgatherv(world, local_buffer, mpi_dtype):
	'''
	If the local buffer is one-dimensional, perform
	a 1-dimensional all-gather. If two-dimensional,
	the resulting matrices are stacked on top of
	each other, e.g. allgather(A1, A2) = [A1 A2].T.
	Could generalize to handle higher-dimensional
	tensors, but this is fine for now.
	'''
	if local_buffer.ndim != 1 and local_buffer.ndim != 2:
		print("Input buffer must be 1 or 2-dimensional")
		exit(1)

	sendcount = np.array([local_buffer.shape[0]], dtype=np.uint64)
	sendcounts = np.empty(world.Get_size(), dtype=np.uint64) 

	world.Allgather([sendcount, MPI.UINT64_T], 
			[sendcounts, MPI.UINT64_T])

	offsets = np.empty(len(sendcounts), dtype=np.uint64)
	offsets[0] = 0
	offsets[1:] = np.cumsum(sendcounts[:-1])

	if local_buffer.ndim == 1:
		shape = np.sum(sendcounts)
	elif local_buffer.ndim == 2:
		cols = local_buffer.shape[1]
		shape = (np.sum(sendcounts), cols)
		sendcounts *= cols
		offsets *= cols 

	recv_buffer = np.empty(shape, dtype=local_buffer.dtype)
	world.Allgatherv([local_buffer, mpi_dtype], 
		[recv_buffer, sendcounts, offsets, mpi_dtype]	
		)

	return recv_buffer

type_to_str = {
	np.uint32: "u32",
	np.uint64: "u64",
	np.float: "float",
	np.double: "double"
} 

str_to_type = {
	"u32": np.uint32,
	"u64": np.uint64,
	"float": np.float,
	"double": np.double 
} 

def get_templated_function(mod, basename, dtypes):
	dtypes_joined = '_'.join([type_to_str[el] for el in dtypes])
	fname = f'{basename}_{dtypes_joined}'
	return getattr(mod, fname) 