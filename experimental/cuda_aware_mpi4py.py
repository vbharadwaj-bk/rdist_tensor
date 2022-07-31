from mpi4py import MPI
import cupy as cp
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
print("starting reduce")
sendbuf = cp.arange(10, dtype='i')
recvbuf = cp.empty_like(sendbuf)
print("rank:", rank, "sendbuff:", sendbuf)
print("rank:", rank, "recvbuff:", recvbuf)
assert hasattr(sendbuf, '__cuda_array_interface__')
assert hasattr(recvbuf, '__cuda_array_interface__')
comm.Allreduce(sendbuf, recvbuf)
print("finished reduce")
print("rank:", rank, "sendbuff:", sendbuf)
print("rank:", rank, "recvbuff:", recvbuf)
assert cp.allclose(recvbuf, sendbuf*size)