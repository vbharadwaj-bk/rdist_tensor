import os, subprocess, tempfile
from multiprocessing import Process
from multiprocessing.connection import Client, Listener
import time

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
        self.exafac = Process(target=exafac_start_function, args=(1, 4))
        self.exafac.start()

        ctrl_hostname, _ = run_shell_cmd("hostname") 
        ctrl_hostname = ctrl_hostname.rstrip("\n")

        self.listener = Listener((ctrl_hostname, RPC_IN_PORT))
        self.from_exafac = self.listener.accept()
        root_hostname = self.from_exafac.recv()
        print(f"Exafac rank 0 is running on {root_hostname}...")
        time.sleep(1)
        self.to_exafac = Client((root_hostname, RPC_OUT_PORT))

    def send_commands(self):
        for i in range(4):
            self.to_exafac.send(["execute", 50])

if __name__=="__main__":
    cpals = Exafac()
    cpals.send_commands()
    