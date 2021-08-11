# Search structure

import consts

import json
import multiprocessing as mp
import os
import socket
import subprocess as sp
import time

class Search:
    def __init__(self, modelpath: str, port=consts.WORKER_PORT, nprocs: int = mp.cpu_count() // 2):
        # Build searcher service, synchronously
        self.sproc = sp.Popen([
            'cargo',
            'build',
            '--release',
        ], cwd=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'search'))

        if self.sproc.wait() != 0:
            raise RuntimeError('Search build failed!')

        # Start searcher service

        self.sproc = sp.Popen([
            'cargo',
            'run',
            '--release',
            '--',
            str(port),
        ], cwd=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'search'))

        # Connect to service

        ok = False
        for _ in range(consts.MAX_RETRIES):
            try:
                self.conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP)
                self.conn.connect(('127.0.0.1', port))
                ok = True
                break
            except ConnectionRefusedError:
                print('Connection refused, retrying..')
                time.sleep(consts.RETRY_DELAY)

        if not ok:
            raise RuntimeError("Couldn't establish connection with search service after {} retries.".format(consts.MAX_RETRIES))

        self.connfile = self.conn.makefile('rw')

        # Write configuration

        config = {
            'Config': {
                'puct_policy_weight': consts.PUCT_POLICY_WEIGHT,
                'puct_noise_weight': consts.PUCT_NOISE_WEIGHT,
                'puct_noise_alpha': consts.PUCT_NOISE_ALPHA,
                'batch_size': consts.BATCH_SIZE,
                'num_threads': nprocs,
                'model_path': modelpath,
                'search_nodes': consts.SEARCH_NODES,
                'temperature': consts.TREE_TEMPERATURE,
            }
        }

        self.write(json.dumps(config))

    def write(self, msg):
        self.connfile.write(msg + '\n')
        self.connfile.flush()

    def readline(self):
        return self.connfile.readline()

    def reset(self):
        self.write('{"Load":[]}')

    def stop(self):
        """Stops the search service and waits for the process to join."""
        self.write('"Stop"')
        self.sproc.wait()

    def go(self, status_data):
        """Starts the search and waits for status data. Returns the final
           status response when the search is complete."""

        self.write('"Go"')
        
        while True:
            resp = json.loads(self.readline())
            
            if 'Searching' in resp:
                status_data(resp['Searching'])
            
            if 'Done' in resp:
                return resp['Done']

    def push(self, action):
        self.write('{"Push":"%s"}' % action)