import os, subprocess, tempfile
from multiprocessing import Process
from multiprocessing.connection import Client, Listener
import time
import argparse

from rpc_utilities import *

RPC_OUT_PORT=18633
RPC_IN_PORT=18632

def exafac_start_function(node_count, proc_count, isolate_controller=False):
    nodes = parse_nodelist(os.environ['SLURM_NODELIST'])
    ctrl_hostname, _ = run_shell_cmd("hostname") 
    ctrl_hostname = ctrl_hostname.rstrip("\n")

    if isolate_controller:
        available_nodes = nodes[1:]
    else:
        available_nodes = nodes
    if node_count > len(available_nodes):
        print("Error, asked for more nodes than allocation supports!") 

    nodelist = ','.join(available_nodes[:node_count])

    stdout, stderr = run_shell_cmd(
        f"srun --nodelist {nodelist} -n {proc_count} python exafac_server.py {ctrl_hostname} {RPC_IN_PORT} {RPC_OUT_PORT}"
    )
    print(stdout)
    print(stderr)

class Exafac:
    def __init__(self):
        self.exafac = Process(target=exafac_start_function, args=(1, 64))
        self.exafac.start()

        ctrl_hostname, _ = run_shell_cmd("hostname") 
        ctrl_hostname = ctrl_hostname.rstrip("\n")

        self.listener = Listener((ctrl_hostname, RPC_IN_PORT))
        self.from_exafac = self.listener.accept()
        root_hostname = self.from_exafac.recv()
        print(f"Exafac rank 0 is running on {root_hostname}...")
        time.sleep(1)
        self.to_exafac = Client((root_hostname, RPC_OUT_PORT))

    def send_command(self, type, payload):
        self.to_exafac.send({"type": type, "payload" : payload})

def parse_argument_list():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input', type=str, help='HDF5 of Input Tensor', required=True)
    parser.add_argument("-t", "--trank", help="Rank of the target decomposition", required=True, type=str)
    parser.add_argument("-s", "--samples", help="Number of samples taken from the KRP", required=False, type=str)
    parser.add_argument("-iter", help="Number of ALS iterations", required=True, type=int)
    parser.add_argument("-rs", help="Random seed", required=False, type=int, default=42)
    parser.add_argument("-o", "--output", help="Output file to print benchmark statistics", required=True)
    parser.add_argument('-g','--grid', type=str, help='Grid Shape (Comma separated)', required=True)
    parser.add_argument('-op','--optimizer', type=str, help='Optimizer to use for tensor decomposition', required=False, default='exact')
    parser.add_argument("-f", "--factor_file", help="File to print the output factors", required=False, type=str)
    parser.add_argument("-p", "--preprocessing", help="Preprocessing algorithm to apply to the tensor", required=False, type=str)
    parser.add_argument("-e", "--epoch_iter", help="Number of iterations per accuracy evaluation epoch", required=False, type=int, default=5)
    parser.add_argument("--reuse", help="Whether or not to reuse samples between optimization rounds", action=argparse.BooleanOptionalAction)
    parser.add_argument("-pre_optim", help="# of Exact ALS iterations to run before sketching (for CAIDA tensors)", required=False, type=int, default=0)
    
    return parser.parse_args()

if __name__=="__main__":
    args = parse_argument_list()
    cpals = Exafac()
    cpals.send_command("initialize", args)
    cpals.send_command("terminate", args)

    sample_count = None
    cpals.send_command("run_iteration", {"samples": sample_count})
    cpals.send_command("compute_fit", {})