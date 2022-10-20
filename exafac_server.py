# Requires rpyc as a dependency
import rpyc
from multiprocessing import Pool, Queue, Lock, Process
from rpyc.utils.server import OneShotServer 
from rpc_utilities import *
import sys
import subprocess

class Exafac_Server(rpyc.Service):
    def on_connect(self, conn):
        # code that runs when a connection is created
        # (to init the service, if needed)
        pass

    def on_disconnect(self, conn):
        # code that runs after the connection has already closed
        # (to finalize the service, if needed)
        pass

    def exposed_get_answer(self): # this is an exposed method
        return 42

    exposed_the_real_answer_though = 43     # an exposed attribute

    def get_question(self):  # while this method is not exposed
        return "what is the airspeed velocity of an unladen swallow?"

if __name__ == "__main__":
    ctrl_hostname, ctrl_port, server_port = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    hostname, _ = run_shell_cmd("hostname") 
    hostname = hostname.rstrip("\n")

    c = rpyc.connect(ctrl_hostname, ctrl_port)
    c.root.set_root_hostname(hostname)
    c.close()

    t = OneShotServer(Exafac_Server, hostname=hostname, port=int(server_port)) 
    t.start()