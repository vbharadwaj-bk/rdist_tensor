import os, subprocess, tempfile
from multiprocessing import Process 
from rpc_utilities import *
import rpyc
from rpyc.utils.server import OneShotServer 
import time

RPC_OUT_PORT=18633
RPC_IN_PORT=18632

class Controller_Service(rpyc.Service):
    def __init__(self):
        self.root_hostname = None

    def on_connect(self, conn):
        pass

    def on_disconnect(self, conn):
        pass

    def exposed_set_root_hostname(self, hostname): 
        print(f"Root is running on {hostname}")
        self.root_hostname = hostname

def start_controller_service():
    ctrl_hostname, _ = run_shell_cmd("hostname") 
    ctrl_hostname = ctrl_hostname.rstrip("\n")

    controller = Controller_Service()
    t = OneShotServer(controller, hostname=ctrl_hostname, port=RPC_IN_PORT)
    t.start()

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
        self.controller_service = Process(target=start_controller_service)
        self.exafac = Process(target=exafac_start_function, args=(1, 1))

        self.controller_service.start()
        time.sleep(1) 
        self.exafac.start()
        self.controller_service.join()

        #c = rpyc.connect("host", port)
        #c.root
        #print(c.root.get_answer())
        #c.close()
        #self.server.join()

if __name__=="__main__":
    cpals = Exafac()
