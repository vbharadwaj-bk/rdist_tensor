# Requires rpyc as a dependency
from multiprocessing.connection import Client, Listener
from re import I

from rpc_utilities import *
import sys
from mpi4py import MPI

class Exafac_Server:
    def __init__(self, ctrl_hostname, ctrl_port, server_port):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        if rank == 0:
            hostname, _ = run_shell_cmd("hostname") 
            hostname = hostname.rstrip("\n")

            self.to_controller = Client((ctrl_hostname, ctrl_port))
            self.to_controller.send(hostname)
            self.listener = Listener((hostname, server_port))
            self.from_controller = self.listener.accept()
 
            #cmd = self.from_controller.recv()

        comm.Barrier()
        for i in range(4):
            if rank == 0:
                cmd = self.from_controller.recv() 
            else:
                cmd = None
            cmd = comm.bcast(cmd, root=0)
            print(f"Rank {rank} got command {cmd}")

if __name__ == "__main__":
    server = Exafac_Server(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))