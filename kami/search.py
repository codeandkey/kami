# Search structure

import consts
from tree import Tree

import json
import multiprocessing as mp
import os
import select
import socket
import subprocess as sp
import sys
import time
import threading

class Search:
    def __init__(self, modelpath: str, port=consts.WORKER_PORT, nprocs: int = mp.cpu_count() // 2):
        self.streams = []
        self.procs   = []

        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP)
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.bind(('0.0.0.0', port))
        self.listener.listen()

        # Spawn workers and wait for connections
        for _ in range(nprocs):
            self.procs.append(
                sp.Popen([
                    'python',
                    'worker.py',
                    str(port),
                    modelpath,
                ], cwd=os.path.dirname(__file__))
            )

        for _ in range(nprocs):
            conn, _ = self.listener.accept()
            self.streams.append(conn)

        print('Search ready.')
        print('Using {} compute workers.'.format(nprocs))

    def write(self, stream, msg: str):
        stream.send(len(msg).to_bytes(4, 'little'))
        stream.send(msg.encode('utf-8'))

    def run(self, tree: Tree, maxnodes=consts.SEARCH_NODES, status_callback=None):
        """Runs a synchronous search on `tree` until `maxnodes` have been searched."""

        starttime = time.time()

        nnodes = 0
        nbatches = 0
        while nnodes < maxnodes:
            for c in self.streams:
                try:
                    ready, _, _ = select.select([c], [], [], 0)

                    if c not in ready:
                        continue

                    def nextbytes(n):
                        bout = b''
                        while len(bout) < n:
                            bout += c.recv(n - len(bout))
                        return bout

                    def recvnext():
                        msglen = int.from_bytes(nextbytes(4), 'little')
                        return nextbytes(msglen).decode('utf-8')

                    msg = recvnext()

                    if msg == 'READY':
                        # Send a batch!
                        nextbatch = tree.next_batch().into_dict()
                        self.write(c, json.dumps(nextbatch))
                    else:
                        # Try and expand batch results.
                        results = json.loads(msg)
                        tree.expand(results)

                        if nnodes == 0:
                            starttime = time.time()

                        nnodes += len(results['value'])
                        nbatches += 1

                        if status_callback:
                            status_callback({
                                'nps': nnodes / (time.time() - starttime),
                                'bps': nbatches / (time.time() - starttime),
                                'nodes': nnodes,
                                'elapsed': (time.time() - starttime),
                                'batches': nbatches,
                            })
                except ConnectionResetError:
                    c.close()
                    self.streams.remove(c)
        
    def stop(self):
        """Stops the search. Sends a STOP request to all workers and waits
           for their processes to terminate."""

        for child in self.streams:
            self.write(child, 'STOP')

        for proc in self.procs:
            proc.wait(5000)